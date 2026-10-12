"""Async path: Router.acomplete, async providers, sync-provider bridging.

No pytest-asyncio: each test drives the coroutine with ``asyncio.run`` so
the suite keeps its single dev dependency (pytest itself). Adapter tests
reuse the local stub-server pattern from test_adapters.py — no network,
no API keys.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from budgetrouter import (
    AsyncFakeProvider,
    BudgetExceeded,
    BudgetGate,
    Completion,
    FakeProvider,
    OpenAICompatibleProvider,
    ProviderError,
    Route,
    RouteRule,
    Router,
    Usage,
)

MESSAGES = [{"role": "user", "content": "hello"}]


def run(coro):
    return asyncio.run(coro)


def make_router(**overrides):
    providers = {
        "fake": FakeProvider({"fake-small": "small answer", "fake-large": "large answer"}),
    }
    kwargs = dict(
        providers=providers,
        rules=[RouteRule(provider="fake", model="fake-large", tag="hard")],
        default=Route(provider="fake", model="fake-small"),
    )
    kwargs.update(overrides)
    return Router(**kwargs)


def test_acomplete_default_route():
    out = run(make_router().acomplete(MESSAGES))
    assert out.text == "small answer"
    assert out.model == "fake-small"


def test_acomplete_tag_rule():
    out = run(make_router().acomplete(MESSAGES, tag="hard"))
    assert out.text == "large answer"
    assert out.model == "fake-large"


def test_acomplete_fallback_on_provider_failure():
    flaky = FakeProvider("primary answer", name="flaky", fail_times=1)
    backup = FakeProvider("backup answer", name="backup")
    router = Router(
        providers={"flaky": flaky, "backup": backup},
        default=Route(provider="flaky", model="fake-small"),
        fallbacks=[Route(provider="backup", model="fake-small")],
    )
    out = run(router.acomplete(MESSAGES))
    assert out.text == "backup answer"
    assert out.provider == "backup"


def test_acomplete_all_routes_failing_raises():
    router = Router(
        providers={
            "a": FakeProvider(name="a", fail_times=5),
            "b": FakeProvider(name="b", fail_times=5),
        },
        default=Route(provider="a", model="fake-small"),
        fallbacks=[Route(provider="b", model="fake-small")],
    )
    with pytest.raises(ProviderError, match="all routes failed"):
        run(router.acomplete(MESSAGES))


def test_acomplete_budget_exceeded_is_not_a_provider_failure():
    backup = FakeProvider("backup answer", name="backup")
    router = Router(
        providers={"fake": FakeProvider("x"), "backup": backup},
        default=Route(provider="fake", model="fake-large"),
        fallbacks=[Route(provider="backup", model="fake-small")],
        budget=BudgetGate(budget_usd=0.0001),
    )
    with pytest.raises(BudgetExceeded):
        run(router.acomplete(MESSAGES))
    assert backup.calls == []


class SyncOnlyProvider:
    """A provider with no acomplete: the router must bridge it via a thread."""

    name = "sync-only"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, model):
        self.calls += 1
        return Completion(
            text="sync answer",
            model=model,
            provider=self.name,
            usage=Usage(input_tokens=3, output_tokens=2),
        )


def test_acomplete_bridges_sync_only_provider():
    provider = SyncOnlyProvider()
    router = Router(
        providers={"sync-only": provider},
        default=Route(provider="sync-only", model="any-model"),
    )
    out = run(router.acomplete(MESSAGES))
    assert out.text == "sync answer"
    assert provider.calls == 1


class StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        payload = json.dumps(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": "stub async answer"}}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def stub_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


def test_adapter_acomplete_against_stub(stub_url):
    provider = OpenAICompatibleProvider(stub_url, "stub-small", api_key="test-key")
    out = run(provider.acomplete(MESSAGES, "stub-small"))
    assert out.text == "stub async answer"
    assert out.usage.input_tokens == 100
    assert out.usage.output_tokens == 50


def test_router_acomplete_with_real_adapter_and_budget(stub_url):
    provider = OpenAICompatibleProvider(stub_url, "stub-small", api_key="k")
    gate = BudgetGate(budget_usd=1.00, prices={"stub-small": (0.001, 0.002)})
    router = Router(
        providers={"openai": provider},
        default=Route(provider="openai", model="stub-small"),
        budget=gate,
    )
    out = run(router.acomplete(MESSAGES))
    assert out.text == "stub async answer"
    assert gate.spent > 0  # usage from the async call was recorded


def test_acomplete_budget_race_is_serialized_by_the_gate():
    # Estimates equal actuals (48 chars -> 12 input tokens, 8 expected
    # output tokens); the cap fits exactly 3 of 10 concurrent calls. The
    # provider sleeps, so without the gate's hold every call would pass
    # the pre-call check before any of them records its spend.
    prices = {"fake-small": (0.001, 0.002)}
    per_call = 12 / 1000 * 0.001 + 8 / 1000 * 0.002
    gate = BudgetGate(budget_usd=per_call * 3.5, prices=prices)
    provider = AsyncFakeProvider(
        "answer", input_tokens=12, output_tokens=8, delay=0.05
    )
    router = Router(
        providers={"fake": provider},
        default=Route(provider="fake", model="fake-small"),
        budget=gate,
        expected_output_tokens=8,
    )
    messages = [{"role": "user", "content": "x" * 48}]

    async def race():
        return await asyncio.gather(
            *(router.acomplete(messages) for _ in range(10)),
            return_exceptions=True,
        )

    results = run(race())
    successes = [r for r in results if isinstance(r, Completion)]
    failures = [r for r in results if isinstance(r, Exception)]
    assert len(successes) == 3
    assert len(failures) == 7
    assert all(isinstance(f, BudgetExceeded) for f in failures)
    assert gate.spent == pytest.approx(per_call * 3)
    assert len(provider.calls) == 3  # blocked calls never reached the provider
