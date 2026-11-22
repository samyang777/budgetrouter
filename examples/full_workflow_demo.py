"""End-to-end demo: failure-aware fallback, bucketed budgets, usage summary, and eval gate.

Runs completely offline with ``FakeProvider`` — no API keys, no network:

    python examples/full_workflow_demo.py
"""

from __future__ import annotations

from budgetrouter import (
    BudgetExceeded,
    BudgetManager,
    Completion,
    FakeProvider,
    FallbackPolicy,
    ProviderError,
    Route,
    RouteRule,
    Router,
    Usage,
    check_gate,
    run_router_eval,
)


def main() -> None:
    # 1. Bucketed budgets with an 80% warning callback
    warnings: list[str] = []

    def on_warning(bucket: str, spent: float, budget_usd: float) -> None:
        msg = f"[WARN] bucket={bucket!r} reached ${spent:.6f} / ${budget_usd:.4f}"
        warnings.append(msg)
        print(msg)

    manager = BudgetManager(
        budgets={"team-search": 0.00018, "team-core": 0.020},
        default_budget_usd=0.010,
        warning_threshold=0.8,
        on_warning=on_warning,
    )

    # 2. Failure-aware fallback: primary times out once -> backup serves;
    #    bad_auth fails with 401 -> aborted immediately without fallback.
    providers = {
        "primary": FakeProvider(
            {"fake-large": "strong answer"},
            name="primary",
            input_tokens=100,
            output_tokens=60,
            fail_times=1,
            fail_kind="timeout",
        ),
        "backup": FakeProvider(
            {"fake-small": "backup answer"},
            name="backup",
            input_tokens=100,
            output_tokens=60,
        ),
        "broken-auth": FakeProvider(
            name="broken-auth",
            fail_times=5,
            fail_kind="auth",
            fail_status_code=401,
        ),
    }

    router = Router(
        providers=providers,
        rules=[
            RouteRule(provider="primary", model="fake-large", tag="hard"),
            RouteRule(provider="broken-auth", model="fake-large", tag="auth-test"),
        ],
        default=Route(provider="backup", model="fake-small"),
        fallbacks=[Route(provider="backup", model="fake-small")],
        budget=manager,
        fallback_policy=FallbackPolicy(),  # auth -> abort; timeout/429/5xx -> fallback
        expected_output_tokens=60,
    )

    # Call 1: primary times out -> visible downgrade to backup
    r1 = router.complete(
        [{"role": "user", "content": "Analyze this incident."}],
        tag="hard",
        bucket="team-core",
    )
    print(
        f"call 1 -> {r1.provider}/{r1.model} "
        f"(attempts={[a.outcome for a in r1.attempts]})"
    )

    # Call 2: primary recovered -> served directly
    r2 = router.complete(
        [{"role": "user", "content": "Draft architecture notes."}],
        tag="hard",
        bucket="team-core",
    )
    print(
        f"call 2 -> {r2.provider}/{r2.model} "
        f"(attempts={[a.outcome for a in r2.attempts]})"
    )

    # Call 3: auth failure aborts immediately without falling back
    try:
        router.complete(
            [{"role": "user", "content": "Check credentials."}],
            tag="auth-test",
            bucket="team-core",
        )
    except ProviderError as exc:
        print(
            f"call 3 -> aborted ({exc.kind}, status={exc.status_code}, "
            f"attempts={len(exc.attempts)})"
        )

    # Call 4+: spend inside team-search until it hits its own cap (team-core unaffected)
    for idx in range(1, 5):
        try:
            router.complete(
                [{"role": "user", "content": f"Search summary #{idx}"}],
                bucket="team-search",
            )
        except BudgetExceeded as exc:
            print(f"team-search capped on call #{idx} (bucket={exc.bucket})")
            break

    # 3. Usage attribution summary
    print("\n--- Usage Summary (bucket, model) ---")
    print(router.usage_summary().format_table())

    # 4. Eval regression gate with cost + quality on the same record
    class DemoEvalProvider:
        name = "eval-mock"

        def complete(self, messages, model):
            q = messages[-1]["content"]
            if model == "fake-small" and "even" in q:
                ans = "I am not sure."
            elif "France" in q:
                ans = "Paris"
            else:
                ans = "2k + 2m = 2(k + m)"
            return Completion(
                text=ans,
                model=model,
                provider=self.name,
                usage=Usage(input_tokens=20, output_tokens=10),
            )

    cases = [
        {"input": "Capital of France?", "target": "Paris", "tag": "easy"},
        {"input": "Sum of two even integers?", "target": "2(", "tag": "hard"},
    ]
    eval_router = Router(
        providers={"eval-mock": DemoEvalProvider()},
        rules=[RouteRule(provider="eval-mock", model="fake-large", tag="hard")],
        default=Route(provider="eval-mock", model="fake-small"),
    )
    report = run_router_eval(cases, eval_router)
    gate = check_gate(report, min_pass_rate=0.90, max_error_rate=0.05)
    print(
        f"\n--- Eval Gate ---\n{report.summary()} | "
        f"total_cost=${report.total_cost_usd:.6f} | {gate.summary()}"
    )


if __name__ == "__main__":
    main()
