"""Live smoke test against a real OpenAI-compatible endpoint.

The unit/integration tests run against a local stub. This script is the
counterpart for a real service: point it at any OpenAI-compatible API
(OpenAI itself, GitHub Models, Groq, Gemini's compatible endpoint, ...)
with a real key and it verifies the adapter end-to-end, then runs two
calls through Router + BudgetGate and checks that spend gets recorded.

Usage:
    export BUDGETROUTER_LIVE_API_KEY=...        # your key (required)
    export BUDGETROUTER_LIVE_BASE_URL=...      # default: GitHub Models
    export BUDGETROUTER_LIVE_MODEL=...          # default: openai/gpt-4o-mini
    python examples/live_smoke_test.py

Defaults target GitHub Models' free tier (endpoint
https://models.github.ai/inference), which needs a GitHub fine-grained
personal access token with the Models read permission.
"""

import os
import sys

from budgetrouter import BudgetGate, OpenAICompatibleProvider, Route, Router
from budgetrouter.budget import DEFAULT_PRICES

BASE_URL = os.environ.get(
    "BUDGETROUTER_LIVE_BASE_URL", "https://models.github.ai/inference"
)
MODEL = os.environ.get("BUDGETROUTER_LIVE_MODEL", "openai/gpt-4o-mini")
API_KEY = os.environ.get("BUDGETROUTER_LIVE_API_KEY")

if not API_KEY:
    sys.exit("Set BUDGETROUTER_LIVE_API_KEY to a real API key first.")


def main() -> None:
    provider = OpenAICompatibleProvider(
        base_url=BASE_URL, model=MODEL, api_key=API_KEY,
        name="live", timeout=120.0,
    )

    print("--- adapter direct call ---")
    c = provider.complete(
        [{"role": "user", "content": "Reply with exactly this text: budget-ok"}],
        MODEL,
    )
    print("text:", repr(c.text))
    print("usage: input", c.usage.input_tokens, "output", c.usage.output_tokens)
    assert c.usage.total_tokens > 0, "usage was not parsed from the response"

    print("--- router + budget end-to-end ---")
    prices = dict(DEFAULT_PRICES)
    prices.setdefault(MODEL, DEFAULT_PRICES.get("gpt-4o-mini", (0.00015, 0.0006)))
    router = Router(
        providers={"live": provider},
        rules=[],
        default=Route(provider="live", model=MODEL),
        fallbacks=[],
        budget=BudgetGate(budget_usd=0.05, per_request_cap_usd=0.02, prices=prices),
    )
    for q in [
        "What is 2 + 2? Answer with just the number.",
        "What is the capital of France? One word.",
    ]:
        r = router.complete([{"role": "user", "content": q}])
        print(f"  -> {r.text!r} (in {r.usage.input_tokens} / out {r.usage.output_tokens})")
    print(f"spent: ${router.budget.spent:.6f} of $0.05 cap")
    assert router.budget.spent > 0, "spend was not recorded"
    print("LIVE SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
