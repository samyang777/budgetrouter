import asyncio
import time

import pytest

from budgetrouter import (
    AsyncFakeProvider,
    AsyncRouter,
    BudgetExceeded,
    BudgetGate,
    BudgetManager,
    Completion,
    FakeProvider,
    Route,
    Router,
    estimate_cost,
)
from budgetrouter.provider import Usage

PRICES = {"fake-small": (0.001, 0.002), "fake-large": (0.010, 0.020)}


def test_estimate_cost_uses_price_table():
    # 1000 input @ $0.001/1K + 500 output @ $0.002/1K = $0.002
    assert estimate_cost("fake-small", 1000, 500, PRICES) == pytest.approx(0.002)


def test_estimate_cost_unknown_model():
    with pytest.raises(KeyError, match="no price configured"):
        estimate_cost("mystery-model", 10, 10, PRICES)


def test_check_allows_call_under_budget():
    gate = BudgetGate(budget_usd=1.00, prices=PRICES)
    est = gate.check("fake-small", est_input_tokens=100, est_output_tokens=50)
    assert est == pytest.approx(0.0002)
    assert gate.spent == 0.0  # check alone does not spend


def test_check_blocks_call_over_budget():
    gate = BudgetGate(budget_usd=0.001, prices=PRICES)
    with pytest.raises(BudgetExceeded, match="run budget"):
        gate.check("fake-large", est_input_tokens=1000, est_output_tokens=1000)


def test_per_request_cap_blocks_expensive_single_call():
    gate = BudgetGate(budget_usd=100.00, per_request_cap_usd=0.005, prices=PRICES)
    with pytest.raises(BudgetExceeded, match="per-request cap"):
        gate.check("fake-large", est_input_tokens=1000, est_output_tokens=0)
    # A cheap call still passes under the same gate.
    gate.check("fake-small", est_input_tokens=100, est_output_tokens=100)


def test_record_accumulates_spend_and_remaining():
    gate = BudgetGate(budget_usd=0.01, prices=PRICES)
    cost = gate.record("fake-small", Usage(input_tokens=1000, output_tokens=1000))
    assert cost == pytest.approx(0.003)
    assert gate.spent == pytest.approx(0.003)
    assert gate.remaining == pytest.approx(0.007)


def test_usage_rejects_negative_token_counts():
    # A negative count would subtract from recorded spend if it got through.
    with pytest.raises(ValueError, match="negative"):
        Usage(input_tokens=-1, output_tokens=0)
    with pytest.raises(ValueError, match="negative"):
        Usage(input_tokens=0, output_tokens=-5)


def test_admitted_call_overshoot_is_recorded_then_blocks():
    # Admission is estimate-based: this call's estimate fits the budget,
    # its actual usage does not. The actual cost is recorded truthfully —
    # recorded spend passes the cap by the estimation error — and every
    # later call is refused before it is made.
    gate = BudgetGate(budget_usd=0.001, prices=PRICES)
    gate.check("fake-small", est_input_tokens=100, est_output_tokens=50)
    cost = gate.record("fake-small", Usage(input_tokens=1000, output_tokens=1000))
    assert cost == pytest.approx(0.003)
    assert gate.spent > gate.budget_usd
    with pytest.raises(BudgetExceeded):
        gate.check("fake-small", est_input_tokens=1, est_output_tokens=1)


def test_router_spend_accumulates_across_calls_until_cap():
    provider = FakeProvider("ok", input_tokens=1000, output_tokens=1000)
    # Each recorded call costs 1000/1000*0.001 + 1000/1000*0.002 = $0.003.
    # The pre-call estimate is ~$0.002 (estimated input is ~1 token for "hi",
    # expected output is 1000 tokens), so the third call's check sees
    # $0.006 already spent + $0.002 estimated > $0.0055 budget and refuses.
    gate = BudgetGate(budget_usd=0.0055, prices=PRICES)
    router = Router(
        providers={"fake": provider},
        default=Route(provider="fake", model="fake-small"),
        budget=gate,
        expected_output_tokens=1000,
    )
    messages = [{"role": "user", "content": "hi"}]
    router.complete(messages)  # spent 0.003
    router.complete(messages)  # spent 0.006
    with pytest.raises(BudgetExceeded):
        router.complete(messages)
    assert len(provider.calls) == 2


# ---------------------------------------------------------------------------
# Warning thresholds & BudgetManager (bucketed caps)
# ---------------------------------------------------------------------------


def test_budget_gate_warning_fires_once_at_threshold():
    events: list[tuple[str, float, float]] = []
    gate = BudgetGate(
        budget_usd=0.010,
        prices=PRICES,
        warning_threshold=0.8,
        on_warning=lambda b, s, cap: events.append((b, s, cap)),
    )
    assert gate.warned is False
    # 1st call: $0.003 (30%) -> no warning
    gate.record("fake-small", Usage(input_tokens=1000, output_tokens=1000))
    assert gate.warned is False
    assert events == []
    # 2nd call: $0.006 (60%) -> no warning
    gate.record("fake-small", Usage(input_tokens=1000, output_tokens=1000))
    assert gate.warned is False
    # 3rd call: $0.009 (90% >= 80%) -> fires once
    gate.record("fake-small", Usage(input_tokens=1000, output_tokens=1000))
    assert gate.warned is True
    assert len(events) == 1
    assert events[0][0] == "default"
    assert events[0][1] == pytest.approx(0.009)
    assert events[0][2] == pytest.approx(0.010)
    # 4th call -> does NOT fire again
    gate.record("fake-small", Usage(input_tokens=100, output_tokens=100))
    assert len(events) == 1


def test_budget_manager_isolates_buckets():
    # Each call costs $0.003 actual (1000 in + 1000 out on fake-small).
    provider = FakeProvider("ok", input_tokens=1000, output_tokens=1000)
    manager = BudgetManager(
        budgets={"tenant-a": 0.0055, "tenant-b": 0.015},
        default_budget_usd=0.010,
        prices=PRICES,
    )
    router = Router(
        providers={"fake": provider},
        default=Route(provider="fake", model="fake-small"),
        budget=manager,
        expected_output_tokens=1000,
    )
    msgs = [{"role": "user", "content": "hi"}]
    # Exhaust tenant-a after 2 calls ($0.006 > $0.005)
    router.complete(msgs, bucket="tenant-a")
    router.complete(msgs, bucket="tenant-a")
    with pytest.raises(BudgetExceeded) as exc_info:
        router.complete(msgs, bucket="tenant-a")
    assert exc_info.value.bucket == "tenant-a"

    # tenant-b is completely unaffected
    out = router.complete(msgs, bucket="tenant-b")
    assert out.text == "ok"
    assert manager.spent_for("tenant-b") == pytest.approx(0.003)

    # Call with no bucket lands in "default"
    router.complete(msgs)
    assert manager.spent_for("default") == pytest.approx(0.003)
    assert manager.total_spent == pytest.approx(0.012)


def test_budget_manager_exceeded_does_not_fall_back():
    backup = FakeProvider("backup", name="backup")
    manager = BudgetManager(budgets={"tight": 0.0001}, prices=PRICES)
    router = Router(
        providers={"fake": FakeProvider("x"), "backup": backup},
        default=Route(provider="fake", model="fake-large"),
        fallbacks=[Route(provider="backup", model="fake-small")],
        budget=manager,
    )
    with pytest.raises(BudgetExceeded) as exc_info:
        router.complete([{"role": "user", "content": "hi"}], bucket="tight")
    assert exc_info.value.bucket == "tight"
    assert backup.calls == []


def test_budget_manager_unknown_bucket_policies():
    # Default policy: raise KeyError
    m_raise = BudgetManager(budgets={"a": 1.0}, default_budget_usd=0.5, prices=PRICES)
    with pytest.raises(KeyError, match="unknown budget bucket"):
        m_raise.resolve("typo")

    # Fallback to default bucket
    m_default = BudgetManager(
        budgets={"a": 1.0},
        default_budget_usd=0.5,
        prices=PRICES,
        unknown_bucket="default",
    )
    name, gate = m_default.resolve("anything")
    assert name == "default"
    assert gate.budget_usd == 0.5

    # Auto-create independent bucket
    m_create = BudgetManager(
        budgets={"a": 1.0},
        default_budget_usd=0.25,
        prices=PRICES,
        unknown_bucket="create",
    )
    name, gate = m_create.resolve("new-tenant")
    assert name == "new-tenant"
    assert gate.budget_usd == 0.25
    assert "new-tenant" in m_create.buckets


def test_budget_manager_warning_fires_once_per_bucket():
    events: list[tuple[str, float, float]] = []
    manager = BudgetManager(
        budgets={"a": 0.006, "b": 0.020},
        prices=PRICES,
        warning_threshold=0.5,
        on_warning=lambda b, s, cap: events.append((b, s, cap)),
    )
    manager.record("fake-small", Usage(1000, 1000), bucket="a")  # $0.003 == 50%
    manager.record("fake-small", Usage(1000, 1000), bucket="a")  # $0.006 -> no duplicate
    manager.record("fake-small", Usage(1000, 1000), bucket="b")  # $0.003 < 50%
    assert manager.warned_for("a") is True
    assert manager.warned_for("b") is False
    assert [e[0] for e in events] == ["a"]


def test_budget_manager_async_race_per_bucket_and_cross_bucket_overlap():
    per_call = estimate_cost("fake-small", 12, 8, PRICES)
    cap = per_call * 2.5  # fits exactly 2 calls in bucket "a"
    manager = BudgetManager(
        budgets={"a": cap, "b": 10.0},
        prices=PRICES,
    )
    provider = AsyncFakeProvider(
        "answer", input_tokens=12, output_tokens=8, delay=0.05
    )
    router = AsyncRouter(
        providers={"fake": provider},
        default=Route(provider="fake", model="fake-small"),
        budget=manager,
        expected_output_tokens=8,
    )
    messages = [{"role": "user", "content": "x" * 48}]

    async def race():
        return await asyncio.gather(
            *(router.complete(messages, bucket="a") for _ in range(6)),
            return_exceptions=True,
        )

    results = asyncio.run(race())
    successes = [r for r in results if isinstance(r, Completion)]
    failures = [r for r in results if isinstance(r, BudgetExceeded)]
    assert len(successes) == 2
    assert len(failures) == 4
    assert manager.spent_for("a") == pytest.approx(per_call * 2)
    assert manager.spent_for("a") <= cap

    # Different buckets run concurrently (independent locks)
    slow_provider = AsyncFakeProvider("ok", delay=0.2)
    router2 = AsyncRouter(
        providers={"fake": slow_provider},
        default=Route(provider="fake", model="fake-small"),
        budget=BudgetManager(budgets={"x": 1.0, "y": 1.0}, prices=PRICES),
    )

    async def two_buckets():
        return await asyncio.gather(
            router2.complete(messages, bucket="x"),
            router2.complete(messages, bucket="y"),
        )

    start = time.perf_counter()
    asyncio.run(two_buckets())
    elapsed = time.perf_counter() - start
    assert elapsed < 0.35  # overlapped instead of ~0.40s serialized
