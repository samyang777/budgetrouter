"""Runnable quickstart: routing + fallback + a budget gate, no API keys."""

from budgetrouter import BudgetGate, FakeProvider, Route, RouteRule, Router


def main() -> None:
    providers = {
        "fake": FakeProvider(
            {"fake-small": "4", "fake-large": "4 (with a full proof sketch)"},
        ),
    }
    router = Router(
        providers=providers,
        rules=[RouteRule(provider="fake", model="fake-large", tag="hard")],
        default=Route(provider="fake", model="fake-small"),
        fallbacks=[Route(provider="fake", model="fake-small")],
        budget=BudgetGate(budget_usd=1.00, per_request_cap_usd=0.05),
    )

    easy = router.complete([{"role": "user", "content": "What is 2 + 2?"}])
    print(f"default route -> {easy.provider}/{easy.model}: {easy.text}")

    hard = router.complete(
        [{"role": "user", "content": "What is 2 + 2? Show your work."}], tag="hard"
    )
    print(f"tag='hard'    -> {hard.provider}/{hard.model}: {hard.text}")

    print(f"spent so far: ${router.budget.spent:.6f} of ${router.budget.budget_usd:.2f}")


if __name__ == "__main__":
    main()
