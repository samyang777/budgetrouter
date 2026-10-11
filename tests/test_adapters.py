"""Integration tests: real adapters against a local stub HTTP server.

The stub speaks just enough of the OpenAI chat-completions and Anthropic
messages shapes (including usage blocks) to exercise the adapters, the
router's fallback chain, and the budget gate end to end — no network, no
API keys.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from budgetrouter import (
    AnthropicProvider,
    BudgetExceeded,
    BudgetGate,
    OpenAICompatibleProvider,
    ProviderError,
    Route,
    Router,
    estimate_cost,
)

PRICES = {"stub-small": (0.001, 0.002), "stub-large": (0.010, 0.020)}

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


class StubHandler(BaseHTTPRequestHandler):
    """Records requests on the server; paths under /fail return HTTP 500."""

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


def test_openai_adapter_parses_response(stub):
    base_url, server = stub
    provider = OpenAICompatibleProvider(base_url, "stub-small", api_key="test-key")
    out = provider.complete([{"role": "user", "content": "hi"}], "stub-small")
    assert out.text == "stub openai answer"
    assert out.provider == "openai"
    assert out.usage.input_tokens == 100
    assert out.usage.output_tokens == 50
    request = server.requests[-1]
    assert request["path"] == "/chat/completions"
    assert request["headers"].get("authorization") == "Bearer test-key"
    assert request["body"]["model"] == "stub-small"


def test_anthropic_adapter_parses_response_and_splits_system(stub):
    base_url, server = stub
    provider = AnthropicProvider("stub-large", api_key="test-key", base_url=base_url)
    out = provider.complete(
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ],
        "stub-large",
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
    assert request["body"]["max_tokens"] == 1024


def test_api_key_from_env(stub, monkeypatch):
    base_url, _ = stub
    monkeypatch.setenv("STUB_KEY", "from-env")
    provider = OpenAICompatibleProvider(
        base_url, "stub-small", api_key_env="STUB_KEY"
    )
    out = provider.complete([{"role": "user", "content": "hi"}], "stub-small")
    assert out.text == "stub openai answer"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNSET_KEY", raising=False)
    with pytest.raises(ProviderError, match="no API key"):
        OpenAICompatibleProvider(
            "http://127.0.0.1:9", "stub-small", api_key_env="DEFINITELY_UNSET_KEY"
        )


def test_http_500_raises_provider_error_with_status(stub):
    base_url, _ = stub
    provider = OpenAICompatibleProvider(
        f"{base_url}/fail", "stub-small", api_key="test-key"
    )
    with pytest.raises(ProviderError, match="HTTP 500"):
        provider.complete([{"role": "user", "content": "hi"}], "stub-small")


def test_router_fallback_and_budget_accounting(stub):
    base_url, _ = stub
    providers = {
        "primary": OpenAICompatibleProvider(
            f"{base_url}/fail", "stub-large", api_key="k", name="primary"
        ),
        "backup": OpenAICompatibleProvider(
            base_url, "stub-small", api_key="k", name="backup"
        ),
    }
    gate = BudgetGate(budget_usd=1.00, prices=PRICES)
    router = Router(
        providers=providers,
        default=Route(provider="primary", model="stub-large"),
        fallbacks=[Route(provider="backup", model="stub-small")],
        budget=gate,
    )
    out = router.complete([{"role": "user", "content": "hello"}])
    assert out.text == "stub openai answer"
    assert out.provider == "backup"  # primary 500s, fallback serves
    # The recorded spend is exactly the fallback call's reported usage cost.
    assert gate.spent == pytest.approx(estimate_cost("stub-small", 100, 50, PRICES))
    assert gate.spent > 0


def test_budget_exceeded_with_real_adapters(stub):
    base_url, _ = stub
    providers = {
        "real": OpenAICompatibleProvider(base_url, "stub-large", api_key="k"),
    }
    gate = BudgetGate(budget_usd=0.000001, prices=PRICES)
    router = Router(
        providers=providers,
        default=Route(provider="real", model="stub-large"),
        budget=gate,
    )
    with pytest.raises(BudgetExceeded):
        router.complete([{"role": "user", "content": "hello"}])
    assert gate.spent == 0.0  # blocked before any call was made
