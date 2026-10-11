"""Agent-workload demo: the full stack against local stub servers.

Simulates a small agent run — ten mixed tasks tagged ``cheap`` or
``strong`` — routed through the real OpenAI-compatible adapter to two stub
providers. Partway through, the "strong" provider starts returning HTTP
500 and the fallback chain takes over; a tight budget cap is approached
but not crossed. Everything runs offline with no API keys:

    python examples/agent_workload_demo.py
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from budgetrouter import (
    DEFAULT_PRICES,
    BudgetGate,
    OpenAICompatibleProvider,
    Route,
    RouteRule,
    Router,
    estimate_cost,
)

BUDGET_USD = 0.017
STRONG_SUCCESSES = 2  # the strong provider fails after this many calls

# flavor -> (prompt_tokens, completion_tokens) the stub reports per call
USAGE_BY_FLAVOR = {"cheap": (100, 40), "strong": (500, 250), "backup": (500, 250)}

TASKS: list[tuple[str, str]] = [
    ("cheap", "Summarize the standup notes"),
    ("strong", "Draft the database migration plan"),
    ("cheap", "Classify this support ticket"),
    ("strong", "Review the security design doc"),
    ("cheap", "Extract action items from the thread"),
    ("strong", "Write the incident postmortem"),
    ("cheap", "Translate the release note"),
    ("strong", "Design the caching strategy"),
    ("cheap", "Tag the backlog issues"),
    ("cheap", "Summarize the customer call"),
]


class StubHandler(BaseHTTPRequestHandler):
    """OpenAI-shaped stub; /strong/* starts returning 500 partway through."""

    def log_message(self, *args):  # keep demo output readable
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        flavor = self.path.strip("/").split("/")[0]
        if flavor == "strong":
            self.server.strong_calls += 1
            if self.server.strong_calls > STRONG_SUCCESSES:
                self._reply(500, {"error": "strong provider is down"})
                return
        prompt = body["messages"][-1]["content"]
        in_tokens, out_tokens = USAGE_BY_FLAVOR[flavor]
        self._reply(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"[{flavor}] completed: {prompt}",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": in_tokens,
                    "completion_tokens": out_tokens,
                },
            },
        )

    def _reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    server.strong_calls = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        providers = {
            "cheap": OpenAICompatibleProvider(
                f"{base}/cheap", "gpt-4o-mini", api_key="stub-key", name="cheap"
            ),
            "strong": OpenAICompatibleProvider(
                f"{base}/strong", "gpt-4o", api_key="stub-key", name="strong"
            ),
            "strong-backup": OpenAICompatibleProvider(
                f"{base}/backup", "gpt-4o", api_key="stub-key", name="strong-backup"
            ),
        }
        gate = BudgetGate(budget_usd=BUDGET_USD)
        router = Router(
            providers=providers,
            rules=[RouteRule(provider="strong", model="gpt-4o", tag="strong")],
            default=Route(provider="cheap", model="gpt-4o-mini"),
            fallbacks=[
                Route(provider="strong-backup", model="gpt-4o"),
                Route(provider="cheap", model="gpt-4o-mini"),
            ],
            budget=gate,
        )

        primary_for_tag = {"cheap": "cheap", "strong": "strong"}
        per_model: dict[str, dict[str, int]] = {}
        fallbacks = 0

        print(f"Agent workload demo — budget cap ${BUDGET_USD:.3f}, {len(TASKS)} tasks")
        print("(stub providers on localhost; no network, no API keys)\n")
        for i, (tag, prompt) in enumerate(TASKS, start=1):
            out = router.complete([{"role": "user", "content": prompt}], tag=tag)
            cost = estimate_cost(
                out.model,
                out.usage.input_tokens,
                out.usage.output_tokens,
                DEFAULT_PRICES,
            )
            stats = per_model.setdefault(
                out.model, {"calls": 0, "in": 0, "out": 0, "cost": 0.0}
            )
            stats["calls"] += 1
            stats["in"] += out.usage.input_tokens
            stats["out"] += out.usage.output_tokens
            stats["cost"] += cost
            note = ""
            if out.provider != primary_for_tag[tag]:
                fallbacks += 1
                note = f"  <- fallback (primary '{primary_for_tag[tag]}' failed)"
            print(
                f"{i:>2}. [{tag:<6}] {prompt:<38} -> {out.provider}/{out.model}"
                f"  ${cost:.6f}{note}"
            )

        print("\n--- summary ---")
        print(f"tasks completed : {len(TASKS)} ({fallbacks} served by fallback)")
        print(
            f"total spend     : ${gate.spent:.6f} of ${BUDGET_USD:.3f} cap "
            f"({gate.spent / BUDGET_USD:.0%} used, ${gate.remaining:.6f} left)"
        )
        print("per-model breakdown:")
        for model, s in sorted(per_model.items()):
            print(
                f"  {model:<12} {s['calls']} calls, "
                f"{s['in'] + s['out']} tokens, ${s['cost']:.6f}"
            )
    finally:
        server.shutdown()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
