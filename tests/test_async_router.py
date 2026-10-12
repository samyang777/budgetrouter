"""AsyncRouter and the Async* provider classes.

Mirrors the sync router/adapter tests for the class-based async API:
routing and fallback with AsyncFakeProvider, the urllib-backed async
adapters against a local stub server, budget enforcement under
concurrency (admission serialized through the gate, so racing calls
cannot all pass the same pre-call check), and a
concurrency proof that slow calls overlap instead of serializing.

No pytest-asyncio: coroutines are driven with ``asyncio.run`` so the
suite keeps its single dev dependency (pytest itself).
"""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from budgetrouter import (
    AsyncAnthropicProvider,
    AsyncFakeProvider,
    AsyncOpenAICompatibleProvider,
    AsyncRouter,
    BudgetExceeded,
    BudgetGate,
    Completion,
    ProviderError,
    Route,
    RouteRule,
    Usage,
    estimate_cost,
)

PRICES = {"stub-small": (0.001, 0.002), "stub-large": (0.010, 0.020)}
MESSAGES = [{"role": "user", "content": "hello"}]

OPENAI_BODY = {
    "choices": [
        {"message": {"role": "assistant", "content": "stub openai answer"}}
    ],
    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
}

ANTHROPIC_BODY = {
    "content": [{"type": "text", "text": "stub anthropic answer"}],
    "usage": {"input_tokens": 80, "output_tokens": 40},
}

SLOW_SECONDS = 0.5


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# AsyncFakeProvider + AsyncRouter: routing and fallback
# ---------------------------------------------------------------------------


def make_router(**overrides):
    providers = {
        "fake": AsyncFakeProvider(
            {"fake-small": "small answer", "fake-large": "large answer"}
        ),
    }
    kwargs = dict(
        providers=providers,
        rules=[RouteRule(provider="fake", model="fake-large", tag="hard")],
        default=Route(provider="fake", model="fake-small"),
    )
    kwargs.update(overrides)
    return AsyncRouter(**kwargs)


def test_async_router_default_route():
    out = run(make_router().complete(MESSAGES))
    assert out.text == "small answer"
    assert out.model == "fake-small"


def test_async_router_tag_rule():
    out = run(make_router().complete(MESSAGES, tag="hard"))
    assert out.text == "large answer"
    assert out.model == "fake-large"


def test_async_router_fallback_on_provider_failure():
    flaky = AsyncFakeProvider("primary answer", name="flaky", fail_times=1)
    backup = AsyncFakeProvider("backup answer", name="backup")
    router = AsyncRouter(
        providers={"flaky": flaky, "backup": backup},
        default=Route(provider="flaky", model="fake-small"),
        fallbacks=[Route(provider="backup", model="fake-small")],
    )
    out = run(router.complete(MESSAGES))
    assert out.text == "backup answer"
    assert out.provider == "backup"


def test_async_router_all_routes_failing_raises():
    router = AsyncRouter(
        providers={
            "a": AsyncFakeProvider(name="a", fail_times=5),
            "b": AsyncFakeProvider(name="b", fail_times=5),
        },
        default=Route(provider="a", model="fake-small"),
        fallbacks=[Route(provider="b", model="fake-small")],
    )
    with pytest.raises(ProviderError, match="all routes failed"):
        run(router.complete(MESSAGES))


def test_async_router_budget_exceeded_does_not_fall_back():
    backup = AsyncFakeProvider("backup answer", name="backup")
    router = AsyncRouter(
        providers={"fake": AsyncFakeProvider("x"), "backup": backup},
        default=Route(provider="fake", model="fake-large"),
        fallbacks=[Route(provider="backup", model="fake-small")],
        budget=BudgetGate(budget_usd=0.0001),
    )
    with pytest.raises(BudgetExceeded):
        run(router.complete(MESSAGES))
    assert backup.calls == []


class SyncOnlyProvider:
    """A provider with only sync ``complete``: AsyncRouter must bridge it."""

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


def test_async_router_bridges_sync_only_provider():
    provider = SyncOnlyProvider()
    router = AsyncRouter(
        providers={"sync-only": provider},
        default=Route(provider="sync-only", model="any-model"),
    )
    out = run(router.complete(MESSAGES))
    assert out.text == "sync answer"
    assert provider.calls == 1


# ---------------------------------------------------------------------------
# Async adapters against the stub server
# ---------------------------------------------------------------------------


class StubHandler(BaseHTTPRequestHandler):
    """Same shapes as test_adapters.py; /fail returns 500, /slow delays."""

    def log_message(self, *args):  # keep test output quiet
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        headers = {k.lower(): v for k, v in self.headers.items()}
        self.server.requests.append(
            {"path": self.path, "headers": headers, "body": body}
        )
        if self.path.startswith("/fail"):
            payload = b'{"error": "stub failure"}'
            self.send_response(500)
        elif self.path.endswith("/chat/completions"):
            if self.path.startswith("/slow"):
                time.sleep(SLOW_SECONDS)
            payload = json.dumps(OPENAI_BODY).encode()
            self.send_response(200)
        elif self.path.endswith("/v1/messages"):
            payload = json.dumps(ANTHROPIC_BODY).encode()
            self.send_response(200)
        else:
            payload = b'{"error": "unknown path"}'
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def stub():
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", server
    server.shutdown()
    thread.join(timeout=5)


def test_async_openai_adapter_parses_response(stub):
    base_url, server = stub
    provider = AsyncOpenAICompatibleProvider(base_url, "stub-small", api_key="test-key")
    out = run(provider.complete(MESSAGES, "stub-small"))
    assert out.text == "stub openai answer"
    assert out.provider == "openai"
    assert out.usage.input_tokens == 100
    assert out.usage.output_tokens == 50
    request = server.requests[-1]
    assert request["path"] == "/chat/completions"
    assert request["headers"].get("authorization") == "Bearer test-key"
    assert request["body"]["model"] == "stub-small"


def test_async_anthropic_adapter_parses_response_and_splits_system(stub):
    base_url, server = stub
    provider = AsyncAnthropicProvider("stub-large", api_key="test-key", base_url=base_url)
    out = run(
        provider.complete(
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hi"},
            ],
            "stub-large",
        )
    )
    assert out.text == "stub anthropic answer"
    assert out.usage.input_tokens == 80
    assert out.usage.output_tokens == 40
    request = server.requests[-1]
    assert request["path"] == "/v1/messages"
    assert request["headers"].get("x-api-key") == "test-key"
    assert request["headers"].get("anthropic-version") == "2023-06-01"
    assert request["body"]["system"] == "Be terse."
    assert request["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_async_openai_adapter_500_raises_provider_error(stub):
    base_url, _ = stub
    provider = AsyncOpenAICompatibleProvider(
        f"{base_url}/fail", "stub-small", api_key="test-key"
    )
    with pytest.raises(ProviderError, match="HTTP 500"):
        run(provider.complete(MESSAGES, "stub-small"))


def test_async_anthropic_adapter_500_raises_provider_error(stub):
    base_url, _ = stub
    provider = AsyncAnthropicProvider(
        "stub-large", api_key="test-key", base_url=f"{base_url}/fail"
    )
    with pytest.raises(ProviderError, match="HTTP 500"):
        run(provider.complete(MESSAGES, "stub-large"))


def test_async_router_with_adapter_and_budget(stub):
    base_url, _ = stub
    provider = AsyncOpenAICompatibleProvider(base_url, "stub-small", api_key="k")
    gate = BudgetGate(budget_usd=1.00, prices=PRICES)
    router = AsyncRouter(
        providers={"openai": provider},
        default=Route(provider="openai", model="stub-small"),
        budget=gate,
    )
    out = run(router.complete(MESSAGES))
    assert out.text == "stub openai answer"
    assert gate.spent == pytest.approx(estimate_cost("stub-small", 100, 50, PRICES))
    assert gate.spent > 0


# ---------------------------------------------------------------------------
# Concurrency: slow calls overlap; the budget cap is a hard ceiling
# ---------------------------------------------------------------------------


def test_concurrent_slow_adapter_calls_overlap(stub):
    base_url, _ = stub
    provider = AsyncOpenAICompatibleProvider(
        f"{base_url}/slow", "stub-small", api_key="k"
    )

    async def two_calls():
        return await asyncio.gather(
            provider.complete(MESSAGES, "stub-small"),
            provider.complete(MESSAGES, "stub-small"),
        )

    start = time.perf_counter()
    outs = run(two_calls())
    elapsed = time.perf_counter() - start
    assert [o.text for o in outs] == ["stub openai answer"] * 2
    # Each call sleeps 0.5s server-side; serialized they would take >= 1.0s.
    assert elapsed >= SLOW_SECONDS * 0.9  # the delay is real
    assert elapsed < SLOW_SECONDS * 1.7  # ...but the calls overlapped


def test_concurrent_router_calls_overlap_without_budget():
    provider = AsyncFakeProvider("answer", delay=SLOW_SECONDS)
    router = AsyncRouter(
        providers={"fake": provider},
        default=Route(provider="fake", model="fake-small"),
    )

    async def two_calls():
        return await asyncio.gather(
            router.complete(MESSAGES), router.complete(MESSAGES)
        )

    start = time.perf_counter()
    outs = run(two_calls())
    elapsed = time.perf_counter() - start
    assert [o.text for o in outs] == ["answer"] * 2
    assert elapsed >= SLOW_SECONDS * 0.9
    assert elapsed < SLOW_SECONDS * 1.7


def test_budget_cap_holds_under_concurrent_race():
    # One call's actual cost: 12 input + 8 output tokens at fake-small prices.
    per_call = estimate_cost("fake-small", 12, 8)
    # Cap fits exactly 3 calls, with half a call of slack against float dust.
    cap = per_call * 3.5
    gate = BudgetGate(budget_usd=cap)
    provider = AsyncFakeProvider(
        "answer", input_tokens=12, output_tokens=8
    )
    router = AsyncRouter(
        providers={"fake": provider},
        default=Route(provider="fake", model="fake-small"),
        budget=gate,
        # Make the pre-call estimate equal the actual usage: a 48-char
        # message estimates to exactly 12 input tokens.
        expected_output_tokens=8,
    )
    messages = [{"role": "user", "content": "x" * 48}]

    async def race():
        return await asyncio.gather(
            *(router.complete(messages) for _ in range(10)),
            return_exceptions=True,
        )

    results = run(race())
    successes = [r for r in results if isinstance(r, Completion)]
    failures = [r for r in results if isinstance(r, Exception)]
    assert len(successes) == 3
    assert len(failures) == 7
    assert all(isinstance(f, BudgetExceeded) for f in failures)
    # The invariant that matters here: with estimates equal to actuals,
    # recorded spend never exceeds the cap.
    assert gate.spent == pytest.approx(per_call * 3)
    assert gate.spent <= cap
    assert len(provider.calls) == 3  # blocked calls never reached the provider


def test_shared_gate_serializes_across_router_instances():
    # Two AsyncRouters sharing one gate: the lock lives on the gate, so
    # the race protection is not per-router. Same numbers as the race
    # test above — the cap fits exactly 3 of the 10 concurrent calls —
    # and the provider's delay means unsynchronized checks would all
    # pass before any spend is recorded.
    per_call = estimate_cost("fake-small", 12, 8)
    cap = per_call * 3.5
    gate = BudgetGate(budget_usd=cap)
    provider = AsyncFakeProvider(
        "answer", input_tokens=12, output_tokens=8, delay=0.05
    )
    routers = [
        AsyncRouter(
            providers={"fake": provider},
            default=Route(provider="fake", model="fake-small"),
            budget=gate,
            expected_output_tokens=8,
        )
        for _ in range(2)
    ]
    messages = [{"role": "user", "content": "x" * 48}]

    async def race():
        return await asyncio.gather(
            *(r.complete(messages) for r in routers for _ in range(5)),
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
