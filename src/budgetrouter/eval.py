"""A tiny eval runner: JSONL cases in, pass rates out.

Datasets are JSONL files with one ``{"input": ..., "target": ...}`` object
per line. A callable maps each input to an output string; scoring is exact
match or substring ("includes"). This is deliberately minimal — it exists so
routing or model changes can be smoke-checked for quality regressions next to
their cost, not to replace a real eval framework.

Run the built-in demo with::

    python -m budgetrouter.eval --demo
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .provider import Completion, FakeProvider, Message, Usage
from .router import Route, Router

Scorer = Callable[[str, str], bool]


def score_exact(output: str, target: str) -> bool:
    return output.strip() == target.strip()


def score_includes(output: str, target: str) -> bool:
    return target.strip() in output


SCORERS: dict[str, Scorer] = {"exact": score_exact, "includes": score_includes}


@dataclass
class CaseResult:
    input: str
    target: str
    output: str
    passed: bool


@dataclass
class EvalReport:
    results: list[CaseResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def summary(self) -> str:
        return f"{self.passed}/{self.total} passed ({self.pass_rate:.0%})"


def load_cases(path: str | Path) -> list[dict[str, str]]:
    """Load a JSONL dataset of {"input", "target"} cases."""
    cases: list[dict[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def run_eval(
    cases: list[dict[str, str]],
    fn: Callable[[str], str],
    scorer: str = "includes",
) -> EvalReport:
    """Run ``fn`` over every case and score the outputs."""
    if scorer not in SCORERS:
        raise ValueError(f"unknown scorer {scorer!r}; choose from {sorted(SCORERS)}")
    score = SCORERS[scorer]
    results = []
    for case in cases:
        output = fn(case["input"])
        results.append(
            CaseResult(
                input=case["input"],
                target=case["target"],
                output=output,
                passed=score(output, case["target"]),
            )
        )
    return EvalReport(results=results)


_DEMO_CASES: list[dict[str, str]] = [
    {"input": "What is the capital of France?", "target": "Paris"},
    {"input": "What is 2 + 2?", "target": "4"},
    {"input": "What color is the sky on a clear day?", "target": "blue"},
]

_DEMO_ANSWERS = {
    "What is the capital of France?": "Paris",
    "What is 2 + 2?": "4",
    "What color is the sky on a clear day?": "The sky is blue.",
}


class _AnswerProvider:
    """Minimal Provider implementation answering from a lookup table.

    Doubles as a worked example of how little a real provider has to do:
    a ``name`` attribute and a ``complete`` method.
    """

    name = "answers"

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    def complete(self, messages: list[Message], model: str) -> Completion:
        question = messages[-1].get("content", "") if messages else ""
        return Completion(
            text=self._answers.get(question, "I don't know."),
            model=model,
            provider=self.name,
            usage=Usage(input_tokens=10, output_tokens=5),
        )


def _demo_fn() -> Callable[[str], str]:
    router = Router(
        providers={"answers": _AnswerProvider(_DEMO_ANSWERS)},
        default=Route(provider="answers", model="fake-small"),
    )

    def fn(text: str) -> str:
        return router.complete([{"role": "user", "content": text}]).text

    return fn


def print_report(report: EvalReport) -> None:
    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"[{mark}] input={r.input!r} target={r.target!r} output={r.output!r}")
    print(report.summary())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="budgetrouter.eval", description="Run a tiny budgetrouter eval.")
    parser.add_argument("--data", help="path to a JSONL dataset of {input, target} cases")
    parser.add_argument("--scorer", choices=sorted(SCORERS), default="includes")
    parser.add_argument("--demo", action="store_true", help="run the built-in demo cases")
    args = parser.parse_args(argv)

    if args.demo:
        cases = _DEMO_CASES
        fn = _demo_fn()
    elif args.data:
        cases = load_cases(args.data)
        provider = FakeProvider()  # echo provider: a smoke test of the harness

        def fn(text: str) -> str:
            return provider.complete([{"role": "user", "content": text}], "fake-small").text

    else:
        parser.error("one of --demo or --data is required")

    print_report(run_eval(cases, fn, scorer=args.scorer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
