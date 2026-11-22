"""In-memory usage attribution ledger and summary views.

Every routed attempt (success, provider failure, or budget rejection) appends
an :class:`~budgetrouter.provider.AttemptRecord` to a :class:`UsageLedger`.
Callers can inspect the raw records or aggregate them into a structured
:class:`UsageSummary` grouped by bucket, model, and/or provider, with a
one-screen ASCII table rendering for CLI output and demos.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .provider import AttemptRecord

VALID_GROUP_FIELDS: frozenset[str] = frozenset({"bucket", "model", "provider"})

_GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "bucket_model": ("bucket", "model"),
    "model_bucket": ("model", "bucket"),
    "bucket": ("bucket",),
    "model": ("model",),
    "provider": ("provider",),
}


def _normalize_group_by(group_by: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(group_by, str):
        if group_by in _GROUP_ALIASES:
            return _GROUP_ALIASES[group_by]
        raise ValueError(
            f"unknown group_by {group_by!r}; choose from "
            f"{sorted(_GROUP_ALIASES)} or a tuple of {sorted(VALID_GROUP_FIELDS)}"
        )
    fields = tuple(group_by)
    if not fields:
        raise ValueError("group_by cannot be empty")
    invalid = [f for f in fields if f not in VALID_GROUP_FIELDS]
    if invalid:
        raise ValueError(
            f"unknown group_by fields {invalid!r}; choose from {sorted(VALID_GROUP_FIELDS)}"
        )
    return fields


def _field_value(record: AttemptRecord, field: str) -> str:
    if field == "bucket":
        return record.bucket or "default"
    if field == "model":
        return record.model
    return record.provider


@dataclass(frozen=True)
class GroupSummary:
    """Aggregated attempt metrics for one grouping key (e.g. ``(bucket, model)``)."""

    key: tuple[str, ...]
    calls: int
    succeeded: int
    failed: int
    downgrades: int
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    estimated_cost_usd: float
    unpriced_calls: int
    failure_counts: dict[str, int]
    avg_latency_ms: float

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.calls if self.calls else 0.0


@dataclass(frozen=True)
class UsageSummary:
    """Rollup of all recorded attempts, grouped by :attr:`group_by`."""

    group_by: tuple[str, ...]
    groups: tuple[GroupSummary, ...]
    total_calls: int
    total_succeeded: int
    total_failed: int
    total_downgrades: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float

    @property
    def success_rate(self) -> float:
        return self.total_succeeded / self.total_calls if self.total_calls else 0.0

    def format_table(self) -> str:
        """Render a compact one-screen ASCII summary table."""
        if not self.groups:
            return "No usage recorded."

        headers = list(self.group_by) + [
            "calls",
            "ok%",
            "downgrades",
            "in_tok",
            "out_tok",
            "cost_usd",
            "avg_ms",
        ]
        rows: list[list[str]] = []
        for g in self.groups:
            cost_str = f"${g.total_cost_usd:.6f}"
            if g.unpriced_calls:
                cost_str += "*"
            rows.append(
                list(g.key)
                + [
                    str(g.calls),
                    f"{g.success_rate:.0%}",
                    str(g.downgrades),
                    str(g.input_tokens),
                    str(g.output_tokens),
                    cost_str,
                    f"{g.avg_latency_ms:.1f}",
                ]
            )

        total_label = ["TOTAL"] + [""] * (len(self.group_by) - 1)
        total_row = total_label + [
            str(self.total_calls),
            f"{self.success_rate:.0%}",
            str(self.total_downgrades),
            str(self.total_input_tokens),
            str(self.total_output_tokens),
            f"${self.total_cost_usd:.6f}",
            "-",
        ]

        all_rows = [headers] + rows + [total_row]
        widths = [
            max(len(r[col]) for r in all_rows) for col in range(len(headers))
        ]
        right_align_from = len(self.group_by)

        def _fmt(row: list[str]) -> str:
            parts = []
            for idx, cell in enumerate(row):
                if idx >= right_align_from:
                    parts.append(cell.rjust(widths[idx]))
                else:
                    parts.append(cell.ljust(widths[idx]))
            return "  ".join(parts)

        sep = "  ".join("-" * w for w in widths)
        lines = [_fmt(headers), sep]
        lines.extend(_fmt(r) for r in rows)
        lines.append(sep)
        lines.append(_fmt(total_row))
        if any(g.unpriced_calls for g in self.groups):
            lines.append("* includes succeeded calls with unpriced models")
        return "\n".join(lines)


class UsageLedger:
    """In-memory collector of :class:`AttemptRecord` objects."""

    def __init__(self, *, max_records: int | None = None) -> None:
        if max_records is not None and max_records <= 0:
            raise ValueError(f"max_records must be positive, got {max_records!r}")
        self._max_records = max_records
        self._records: deque[AttemptRecord] = deque(maxlen=max_records)

    def record(self, attempt: AttemptRecord) -> None:
        self._records.append(attempt)

    @property
    def records(self) -> tuple[AttemptRecord, ...]:
        return tuple(self._records)

    def clear(self) -> None:
        self._records.clear()

    def summary(
        self, group_by: str | tuple[str, ...] = ("bucket", "model")
    ) -> UsageSummary:
        fields = _normalize_group_by(group_by)
        buckets: dict[tuple[str, ...], list[AttemptRecord]] = {}
        for rec in self._records:
            key = tuple(_field_value(rec, f) for f in fields)
            buckets.setdefault(key, []).append(rec)

        groups: list[GroupSummary] = []
        for key in sorted(buckets):
            recs = buckets[key]
            calls = len(recs)
            succeeded = sum(1 for r in recs if r.succeeded)
            failed = calls - succeeded
            downgrades = sum(1 for r in recs if r.downgraded)
            in_tok = sum(r.input_tokens for r in recs)
            out_tok = sum(r.output_tokens for r in recs)
            actual_cost = sum(
                r.actual_cost_usd for r in recs if r.actual_cost_usd is not None
            )
            est_cost = sum(
                r.estimated_cost_usd
                for r in recs
                if r.estimated_cost_usd is not None
            )
            unpriced = sum(
                1 for r in recs if r.succeeded and r.actual_cost_usd is None
            )
            failure_counts: dict[str, int] = {}
            for r in recs:
                if not r.succeeded:
                    failure_counts[r.outcome] = failure_counts.get(r.outcome, 0) + 1
            avg_ms = sum(r.latency_ms for r in recs) / calls if calls else 0.0
            groups.append(
                GroupSummary(
                    key=key,
                    calls=calls,
                    succeeded=succeeded,
                    failed=failed,
                    downgrades=downgrades,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    total_cost_usd=actual_cost,
                    estimated_cost_usd=est_cost,
                    unpriced_calls=unpriced,
                    failure_counts=failure_counts,
                    avg_latency_ms=avg_ms,
                )
            )

        total_calls = len(self._records)
        total_succeeded = sum(g.succeeded for g in groups)
        total_failed = sum(g.failed for g in groups)
        total_downgrades = sum(g.downgrades for g in groups)
        total_in = sum(g.input_tokens for g in groups)
        total_out = sum(g.output_tokens for g in groups)
        total_cost = sum(g.total_cost_usd for g in groups)

        return UsageSummary(
            group_by=fields,
            groups=tuple(groups),
            total_calls=total_calls,
            total_succeeded=total_succeeded,
            total_failed=total_failed,
            total_downgrades=total_downgrades,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_cost_usd=total_cost,
        )

    def format_table(
        self, group_by: str | tuple[str, ...] = ("bucket", "model")
    ) -> str:
        return self.summary(group_by=group_by).format_table()
