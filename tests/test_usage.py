"""Tests for Workstream 3: UsageLedger and UsageSummary attribution."""

import asyncio

import pytest

from budgetrouter import (
    AsyncFakeProvider,
    AsyncRouter,
    BudgetExceeded,
    BudgetManager,
    FakeProvider,
    Route,
    RouteRule,
    Router,
    UsageLedger,
    estimate_cost,
)

PRICES = {"fake-small": (0.001, 0.002), "fake-large": (0.010, 0.020)}
MESSAGES = [{"role": "user", "content": "hello"}]


def test_usage_summary_reconciles_per_group_cost_with_budget_manager():
    providers = {
        "fake": FakeProvider(
            {"fake-small": "s", "fake-large": "l"},
            input_tokens=100,
            output_tokens=50,
        )
    }
    manager = BudgetManager(
        budgets={"team-a": 1.0, "team-b": 1.0},
        default_budget_usd=1.0,
        prices=PRICES,
    )
    router = Router(
        providers=providers,
        rules=[RouteRule(provider="fake", model="fake-large", tag="hard")],
        default=Route(provider="fake", model="fake-small"),
        budget=manager,
    )

    router.complete(MESSAGES, bucket="team-a")
    router.complete(MESSAGES, tag="hard", bucket="team-a")
    router.complete(MESSAGES, bucket="team-b")

    summary = router.usage_summary()  # default group_by=("bucket", "model")
    assert summary.total_calls == 3
    assert summary.total_succeeded == 3
    assert summary.total_failed == 0
    assert summary.total_downgrades == 0

    by_key = {g.key: g for g in summary.groups}
    expected_small = estimate_cost("fake-small", 100, 50, PRICES)
    expected_large = estimate_cost("fake-large", 100, 50, PRICES)

    assert by_key[("team-a", "fake-small")].total_cost_usd == pytest.approx(expected_small)
    assert by_key[("team-a", "fake-large")].total_cost_usd == pytest.approx(expected_large)
    assert by_key[("team-b", "fake-small")].total_cost_usd == pytest.approx(expected_small)
    assert summary.total_cost_usd == pytest.approx(manager.total_spent)

    # Alternate grouping by model
    by_model = {g.key: g for g in router.usage_summary(group_by="model").groups}
    assert by_model[("fake-small",)].calls == 2
    assert by_model[("fake-large",)].calls == 1


def test_failed_attempts_and_downgrades_appear_in_summary():
    flaky = FakeProvider(
        "primary",
        name="primary",
        fail_times=1,
        fail_kind="server_error",
        fail_status_code=503,
    )
    backup = FakeProvider("backup", name="backup", input_tokens=50, output_tokens=25)
    manager = BudgetManager(budgets={"t1": 0.0001, "t2": 1.0}, prices=PRICES)
    router = Router(
        providers={"primary": flaky, "backup": backup},
        default=Route(provider="primary", model="fake-large"),
        fallbacks=[Route(provider="backup", model="fake-small")],
        budget=manager,
    )

    # Call on t2: primary fails (server_error), backup serves (downgrade)
    out = router.complete(MESSAGES, bucket="t2")
    assert out.provider == "backup"

    # Call on t1: blocked by budget
    with pytest.raises(BudgetExceeded):
        router.complete(MESSAGES, bucket="t1")

    summary = router.usage_summary()
    by_key = {g.key: g for g in summary.groups}

    # Primary failure: 1 call, 0 succeeded, cost 0.0, failure_counts={"server_error": 1}
    g_primary = by_key[("t2", "fake-large")]
    assert g_primary.calls == 1
    assert g_primary.succeeded == 0
    assert g_primary.failed == 1
    assert g_primary.total_cost_usd == 0.0
    assert g_primary.failure_counts == {"server_error": 1}

    # Backup downgrade: 1 call, 1 succeeded, 1 downgrade, positive cost
    g_backup = by_key[("t2", "fake-small")]
    assert g_backup.calls == 1
    assert g_backup.succeeded == 1
    assert g_backup.downgrades == 1
    assert g_backup.total_cost_usd == pytest.approx(
        estimate_cost("fake-small", 50, 25, PRICES)
    )

    # Budget rejection recorded with 0 actual cost
    g_blocked = by_key[("t1", "fake-large")]
    assert g_blocked.calls == 1
    assert g_blocked.failed == 1
    assert g_blocked.total_cost_usd == 0.0
    assert g_blocked.failure_counts == {"budget_exceeded": 1}

    table = summary.format_table()
    assert "bucket" in table
    assert "model" in table
    assert "TOTAL" in table


def test_unpriced_model_without_budget_does_not_crash():
    router = Router(
        providers={"fake": FakeProvider("ok")},
        default=Route(provider="fake", model="custom-local-model"),
    )
    out = router.complete(MESSAGES)
    assert out.text == "ok"
    summary = router.usage_summary()
    assert summary.total_calls == 1
    assert summary.groups[0].unpriced_calls == 1
    assert summary.total_cost_usd == 0.0
    assert "*" in summary.format_table()


def test_empty_ledger_produces_empty_summary():
    ledger = UsageLedger()
    summary = ledger.summary()
    assert summary.total_calls == 0
    assert summary.success_rate == 0.0
    assert summary.groups == ()
    assert summary.format_table() == "No usage recorded."


def test_async_router_populates_shared_usage_ledger():
    ledger = UsageLedger(max_records=10)
    router = AsyncRouter(
        providers={"fake": AsyncFakeProvider("ok", input_tokens=10, output_tokens=5)},
        default=Route(provider="fake", model="fake-small"),
        usage=ledger,
    )
    asyncio.run(router.complete(MESSAGES, bucket="async-job"))
    summary = ledger.summary()
    assert summary.total_calls == 1
    assert summary.groups[0].key == ("async-job", "fake-small")
    ledger.clear()
    assert ledger.records == ()
