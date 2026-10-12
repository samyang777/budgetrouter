import pytest

from budgetrouter import BudgetExceeded, BudgetGate, FakeProvider, Route, Router, estimate_cost
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
