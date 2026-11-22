import json
from pathlib import Path

import pytest

from budgetrouter import (
    BudgetGate,
    FakeProvider,
    ProviderError,
    Route,
    RouteRule,
    Router,
    check_gate,
    load_cases,
    run_eval,
    run_router_eval,
)
from budgetrouter.eval import main

CASES = [
    {"input": "capital of France?", "target": "Paris"},
    {"input": "capital of Italy?", "target": "Rome"},
    {"input": "capital of Spain?", "target": "Madrid"},
]

ANSWERS = {
    "capital of France?": "Paris",
    "capital of Italy?": "Rome is the capital",
    "capital of Spain?": "I don't know",
}


def test_run_eval_includes_scoring():
    report = run_eval(CASES, ANSWERS.__getitem__, scorer="includes")
    assert report.total == 3
    assert report.passed == 2
    assert report.pass_rate == pytest.approx(2 / 3)
    assert report.summary() == "2/3 passed (67%)"


def test_run_eval_exact_scoring():
    report = run_eval(CASES, ANSWERS.__getitem__, scorer="exact")
    # Only "Paris" matches exactly; "Rome is the capital" does not.
    assert report.passed == 1


def test_run_eval_rejects_unknown_scorer():
    with pytest.raises(ValueError, match="unknown scorer"):
        run_eval(CASES, ANSWERS.__getitem__, scorer="vibes")


def test_load_cases_roundtrip(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in CASES) + "\n", encoding="utf-8")
    assert load_cases(path) == CASES


def test_eval_with_fake_provider():
    provider = FakeProvider({"fake-small": "Paris"})
    report = run_eval(
        [{"input": "capital of France?", "target": "Paris"}],
        lambda text: provider.complete([{"role": "user", "content": text}], "fake-small").text,
    )
    assert report.passed == 1


def test_cli_demo_runs(capsys):
    assert main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "3/3 passed (100%)" in out


def test_check_gate_passes_and_fails_on_pass_rate():
    good = run_eval(CASES, lambda q: {"capital of France?": "Paris", "capital of Italy?": "Rome", "capital of Spain?": "Madrid"}[q])
    gate_ok = check_gate(good, min_pass_rate=0.90, max_error_rate=0.05)
    assert bool(gate_ok) is True
    assert gate_ok.passed is True
    assert gate_ok.reasons == ()

    weak = run_eval(CASES, ANSWERS.__getitem__, scorer="includes")  # 2/3 = 66.7%
    gate_fail = check_gate(weak, min_pass_rate=0.90, max_error_rate=0.05)
    assert bool(gate_fail) is False
    assert gate_fail.passed is False
    assert any("pass_rate" in r for r in gate_fail.reasons)


def test_check_gate_fails_when_error_rate_exceeds_cap_even_if_pass_rate_ok():
    # 9 pass, 1 error -> pass_rate = 90% (>= 80%), error_rate = 10% (> 5%) -> MUST fail
    ten_cases = [{"input": f"q{i}", "target": "ok"} for i in range(10)]

    def flaky(q: str) -> str:
        if q == "q9":
            raise ProviderError("boom", kind="server_error")
        return "ok"

    report = run_eval(ten_cases, flaky)
    assert report.pass_rate == pytest.approx(0.9)
    assert report.errors == 1
    assert report.error_rate == pytest.approx(0.1)

    gate = check_gate(report, min_pass_rate=0.80, max_error_rate=0.05)
    assert gate.passed is False
    assert any("error_rate" in r for r in gate.reasons)


def test_check_gate_fails_on_empty_report():
    empty = run_eval([], lambda x: x)
    gate = check_gate(empty, min_pass_rate=0.0, max_error_rate=1.0)
    assert gate.passed is False
    assert "no cases evaluated" in gate.reasons


def test_run_router_eval_catches_quality_regression_and_records_cost():
    class ModelAwareProvider:
        name = "mock"

        def complete(self, messages, model):
            from budgetrouter import Completion, Usage

            q = messages[-1]["content"]
            # Small model fails the hard case; large model gets both right.
            if model == "fake-small" and "prove" in q.lower():
                text = "unsure"
            elif "France" in q:
                text = "Paris"
            else:
                text = "2k + 2m = 2(k + m)"
            return Completion(
                text=text,
                model=model,
                provider=self.name,
                usage=Usage(input_tokens=20, output_tokens=10),
            )

    cases = [
        {"input": "What is the capital of France?", "target": "Paris", "tag": "easy"},
        {"input": "Prove sum of even integers is even", "target": "2(", "tag": "hard"},
    ]

    # Good routing: hard -> fake-large
    good_router = Router(
        providers={"mock": ModelAwareProvider()},
        rules=[RouteRule(provider="mock", model="fake-large", tag="hard")],
        default=Route(provider="mock", model="fake-small"),
        budget=BudgetGate(budget_usd=1.0),
    )
    good_report = run_router_eval(cases, good_router)
    assert check_gate(good_report, min_pass_rate=1.0, max_error_rate=0.0).passed is True
    assert good_report.total_cost_usd > 0
    assert good_report.results[1].route == "mock/fake-large"
    assert good_report.results[1].cost_usd is not None

    # Bad routing change: forces everything to fake-small -> caught by gate!
    bad_router = Router(
        providers={"mock": ModelAwareProvider()},
        rules=[],
        default=Route(provider="mock", model="fake-small"),
        budget=BudgetGate(budget_usd=1.0),
    )
    bad_report = run_router_eval(cases, bad_router)
    gate = check_gate(bad_report, min_pass_rate=1.0, max_error_rate=0.0)
    assert gate.passed is False
    assert bad_report.pass_rate == pytest.approx(0.5)


def test_cli_gate_exits_zero_on_pass_and_nonzero_on_fail(tmp_path, capsys):
    assert main(["--demo", "--gate", "--min-pass-rate", "1.0"]) == 0
    assert "[GATE PASS]" in capsys.readouterr().out

    # Echo provider returns "echo: <input>" which won't contain target "Paris"
    data = tmp_path / "cases.jsonl"
    data.write_text(json.dumps({"input": "capital of France?", "target": "Paris"}) + "\n")
    assert main(["--data", str(data), "--gate", "--min-pass-rate", "0.9"]) == 1
    assert "[GATE FAIL]" in capsys.readouterr().out


def test_cli_target_loads_custom_callable(tmp_path, monkeypatch, capsys):
    mod_file = tmp_path / "custom_eval_target.py"
    mod_file.write_text("def answer(q: str) -> str:\n    return 'Paris'\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    data = tmp_path / "cases.jsonl"
    data.write_text(json.dumps({"input": "capital?", "target": "Paris"}) + "\n")
    rc = main(
        [
            "--data",
            str(data),
            "--target",
            "custom_eval_target:answer",
            "--gate",
            "--min-pass-rate",
            "1.0",
        ]
    )
    assert rc == 0
    assert "[GATE PASS]" in capsys.readouterr().out


def test_placeholder_golden_file_loads():
    placeholder = Path(__file__).resolve().parents[1] / "eval" / "placeholder_golden.jsonl"
    cases = load_cases(placeholder)
    assert len(cases) >= 3
    assert all("input" in c and "target" in c for c in cases)
