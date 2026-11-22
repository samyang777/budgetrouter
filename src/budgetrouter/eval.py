"""Eval harness and regression gate: JSONL cases in, pass/error/cost metrics out.

Datasets are JSONL files with one ``{"input": ..., "target": ...}`` object
per line (optional ``"tag"`` and ``"bucket"`` keys are honored by
:func:`run_router_eval`). Scoring is exact match or substring (``"includes"``).

:func:`check_gate` turns an :class:`EvalReport` into a deterministic pass/fail
decision against configurable ``min_pass_rate`` and ``max_error_rate``
thresholds. :func:`run_router_eval` runs directly through a :class:`Router`
so cost, route, and downgrade status live on the same :class:`CaseResult`
record as quality.

Run the built-in demo with::

    python -m budgetrouter.eval --demo

Run a regression gate against a custom callable (``module:callable``)::

    python -m budgetrouter.eval --gate --data eval/placeholder_golden.jsonl --target my_app:answer
"""

from __future__ import annotations

import argparse
import importlib
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
    error: str | None = None
    route: str | None = None
    cost_usd: float | None = None
    downgraded: bool = False


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

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    @property
    def error_rate(self) -> float:
        return self.errors / self.total if self.total else 0.0

    @property
    def downgrades(self) -> int:
        return sum(1 for r in self.results if r.downgraded)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results if r.cost_usd is not None)

    def summary(self) -> str:
        base = f"{self.passed}/{self.total} passed ({self.pass_rate:.0%})"
        if self.errors:
            base += f", {self.errors} errors ({self.error_rate:.0%})"
        return base


@dataclass(frozen=True)
class GateResult:
    """Outcome of :func:`check_gate`. Truthy iff :attr:`passed` is ``True``."""

    passed: bool
    pass_rate: float
    error_rate: float
    min_pass_rate: float
    max_error_rate: float
    reasons: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.passed

    def summary(self) -> str:
        if self.passed:
            return (
                f"[GATE PASS] pass_rate={self.pass_rate:.0%} "
                f"(min {self.min_pass_rate:.0%}), "
                f"error_rate={self.error_rate:.0%} "
                f"(max {self.max_error_rate:.0%})"
            )
        return "[GATE FAIL] " + "; ".join(self.reasons)


def check_gate(
    report: EvalReport,
    *,
    min_pass_rate: float = 0.90,
    max_error_rate: float = 0.05,
) -> GateResult:
    """Check whether ``report`` satisfies quality and error-rate thresholds.

    Thresholds are policy parameters with sensible defaults (90% minimum pass
    rate, 5% maximum error rate). An empty report always fails the gate.
    """
    if not (0.0 <= min_pass_rate <= 1.0):
        raise ValueError(
            f"min_pass_rate must be in [0, 1], got {min_pass_rate!r}"
        )
    if not (0.0 <= max_error_rate <= 1.0):
        raise ValueError(
            f"max_error_rate must be in [0, 1], got {max_error_rate!r}"
        )

    reasons: list[str] = []
    if report.total == 0:
        reasons.append("no cases evaluated")
    else:
        if report.pass_rate < min_pass_rate:
            reasons.append(
                f"pass_rate {report.pass_rate:.1%} < min_pass_rate {min_pass_rate:.1%}"
            )
        if report.error_rate > max_error_rate:
            reasons.append(
                f"error_rate {report.error_rate:.1%} > max_error_rate {max_error_rate:.1%}"
            )
    return GateResult(
        passed=not reasons,
        pass_rate=report.pass_rate,
        error_rate=report.error_rate,
        min_pass_rate=min_pass_rate,
        max_error_rate=max_error_rate,
        reasons=tuple(reasons),
    )


def load_cases(path: str | Path) -> list[dict[str, str]]:
    """Load a JSONL dataset of ``{"input", "target"}`` cases."""
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
    *,
    catch_errors: bool = True,
) -> EvalReport:
    """Run ``fn`` over every case and score the outputs."""
    if scorer not in SCORERS:
        raise ValueError(f"unknown scorer {scorer!r}; choose from {sorted(SCORERS)}")
    score = SCORERS[scorer]
    results: list[CaseResult] = []
    for case in cases:
        try:
            output = fn(case["input"])
        except Exception as exc:
            if not catch_errors:
                raise
            results.append(
                CaseResult(
                    input=case["input"],
                    target=case["target"],
                    output="",
                    passed=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        results.append(
            CaseResult(
                input=case["input"],
                target=case["target"],
                output=output,
                passed=score(output, case["target"]),
            )
        )
    return EvalReport(results=results)


def run_router_eval(
    cases: list[dict[str, str]],
    router: Router,
    *,
    scorer: str = "includes",
    tag: str | None = None,
    bucket: str | None = None,
    catch_errors: bool = True,
) -> EvalReport:
    """Evaluate a :class:`Router` so quality, cost, route, and downgrades stay together."""
    if scorer not in SCORERS:
        raise ValueError(f"unknown scorer {scorer!r}; choose from {sorted(SCORERS)}")
    score = SCORERS[scorer]
    results: list[CaseResult] = []
    for case in cases:
        case_tag = case.get("tag", tag)
        case_bucket = case.get("bucket", bucket)
        try:
            completion = router.complete(
                [{"role": "user", "content": case["input"]}],
                tag=case_tag,
                bucket=case_bucket,
            )
        except Exception as exc:
            if not catch_errors:
                raise
            results.append(
                CaseResult(
                    input=case["input"],
                    target=case["target"],
                    output="",
                    passed=False,
                    error=f"{type(exc).__name__}: {exc}",
                    cost_usd=0.0,
                )
            )
            continue
        priced_costs = [
            a.actual_cost_usd
            for a in completion.attempts
            if a.actual_cost_usd is not None
        ]
        cost_usd = sum(priced_costs) if priced_costs else None
        downgraded = any(a.downgraded for a in completion.attempts)
        results.append(
            CaseResult(
                input=case["input"],
                target=case["target"],
                output=completion.text,
                passed=score(completion.text, case["target"]),
                route=f"{completion.provider}/{completion.model}",
                cost_usd=cost_usd,
                downgraded=downgraded,
            )
        )
    return EvalReport(results=results)


def _load_target_callable(spec: str) -> Callable[[str], str]:
    """Load a ``module:attribute`` callable for CLI evaluation."""
    if ":" not in spec:
        raise ValueError(
            f"invalid --target {spec!r}; expected 'module.path:callable_name'"
        )
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)
    if not callable(obj):
        raise TypeError(f"target {spec!r} is not callable")
    return obj


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
        if r.error is not None:
            print(f"[ERR ] input={r.input!r} target={r.target!r} error={r.error!r}")
        else:
            mark = "PASS" if r.passed else "FAIL"
            extra = ""
            if r.route is not None:
                extra += f" route={r.route}"
            if r.cost_usd is not None:
                extra += f" cost=${r.cost_usd:.6f}"
            if r.downgraded:
                extra += " [downgraded]"
            print(
                f"[{mark}] input={r.input!r} target={r.target!r} "
                f"output={r.output!r}{extra}"
            )
    print(report.summary())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="budgetrouter.eval", description="Run a budgetrouter eval or regression gate."
    )
    parser.add_argument(
        "--data", help="path to a JSONL dataset of {input, target} cases"
    )
    parser.add_argument("--scorer", choices=sorted(SCORERS), default="includes")
    parser.add_argument(
        "--demo", action="store_true", help="run the built-in demo cases"
    )
    parser.add_argument(
        "--target",
        help="callable to evaluate as 'module:callable' (used with --data)",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="exit non-zero if the report fails pass/error rate thresholds",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.90,
        help="minimum pass rate for --gate (default: 0.90)",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.05,
        help="maximum error rate for --gate (default: 0.05)",
    )
    args = parser.parse_args(argv)

    if args.demo:
        cases = _DEMO_CASES
        fn = _load_target_callable(args.target) if args.target else _demo_fn()
    elif args.data:
        cases = load_cases(args.data)
        if args.target:
            fn = _load_target_callable(args.target)
        else:
            provider = FakeProvider()  # echo provider: a smoke test of the harness

            def fn(text: str) -> str:
                return provider.complete(
                    [{"role": "user", "content": text}], "fake-small"
                ).text

    else:
        parser.error("one of --demo or --data is required")

    report = run_eval(cases, fn, scorer=args.scorer)
    print_report(report)
    if args.gate:
        gate = check_gate(
            report,
            min_pass_rate=args.min_pass_rate,
            max_error_rate=args.max_error_rate,
        )
        print(gate.summary())
        return 0 if gate.passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
