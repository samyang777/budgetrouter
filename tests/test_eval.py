import json

import pytest

from budgetrouter import FakeProvider, load_cases, run_eval
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
