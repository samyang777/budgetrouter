"""Budget gates: cumulative cost tracking with run, per-request, and per-bucket caps.

Costs are estimated from a simple editable price table of USD per 1K tokens
(input, output) keyed by model name. The gate is consulted *before* a call
(with estimated token counts) and updated *after* it (with actual usage), so
a runaway agent loop hits a clear :class:`BudgetExceeded` error instead of a
surprise bill.

Admission is estimate-based, by necessity: a call's true cost exists only
after the call returns. A call admitted under a cap can therefore push
recorded spend past that cap by its estimation error — but recording is
always truthful, and once recorded spend reaches a cap, every further call
is refused before it is made.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Literal

from .provider import AttemptRecord, Usage

WarningCallback = Callable[[str, float, float], None]
"""Callback invoked once per bucket on warning crossing: ``(bucket, spent, budget_usd)``."""

UnknownBucketPolicy = Literal["raise", "create", "default"]

# USD per 1K tokens: (input, output). Edit freely; models used with a
# BudgetGate must appear here.
#
# The vendor entries below are ILLUSTRATIVE defaults only, roughly in line
# with public list prices when this file was written. Vendor pricing
# changes often — verify the current numbers on the vendor's pricing page
# before relying on them for a real budget.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "fake-small": (0.0002, 0.0006),
    "fake-large": (0.0030, 0.0090),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.0100),
    "claude-sonnet-4-5": (0.0030, 0.0150),
    "claude-haiku-4-5": (0.0010, 0.0050),
}


class BudgetExceeded(RuntimeError):
    """Raised when a call would exceed a configured budget cap."""

    def __init__(
        self,
        message: str = "",
        *,
        bucket: str | None = None,
        attempts: tuple[AttemptRecord, ...] = (),
    ) -> None:
        super().__init__(message)
        self.bucket: str | None = bucket
        self.attempts: tuple[AttemptRecord, ...] = tuple(attempts)


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    prices: dict[str, tuple[float, float]] = DEFAULT_PRICES,
) -> float:
    """Estimated USD cost of one call at the given token counts."""
    if model not in prices:
        raise KeyError(
            f"no price configured for model {model!r}; add it to the price table"
        )
    in_price, out_price = prices[model]
    return (input_tokens / 1000) * in_price + (output_tokens / 1000) * out_price


def _validate_warning_threshold(warning_threshold: float | None) -> None:
    if warning_threshold is not None and not (0.0 < warning_threshold <= 1.0):
        raise ValueError(
            f"warning_threshold must be in (0, 1], got {warning_threshold!r}"
        )


class BudgetGate:
    """Track cumulative spend and refuse calls that would exceed the caps.

    ``budget_usd`` caps the whole run; ``per_request_cap_usd`` (optional)
    caps any single call, which catches accidental calls to a very expensive
    model even when the run budget is still healthy.

    Both caps gate *admission* on the estimated cost (see :meth:`check`);
    :meth:`record` then books the *actual* cost. A single call whose actual
    usage exceeds its estimate can therefore carry recorded spend past a
    cap by that estimation error. What the caps guarantee is that no
    further call is admitted once recorded spend has reached them.

    When ``warning_threshold`` is set (default ``0.8``, i.e. 80%), crossing
    that fraction of ``budget_usd`` during :meth:`record` sets
    :attr:`warned` to ``True`` and invokes ``on_warning(bucket_name, spent,
    budget_usd)`` once.
    """

    def __init__(
        self,
        budget_usd: float,
        *,
        per_request_cap_usd: float | None = None,
        prices: dict[str, tuple[float, float]] | None = None,
        warning_threshold: float | None = 0.8,
        on_warning: WarningCallback | None = None,
        bucket_name: str = "default",
    ) -> None:
        if budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        _validate_warning_threshold(warning_threshold)
        self.budget_usd = budget_usd
        self.per_request_cap_usd = per_request_cap_usd
        self.prices = prices if prices is not None else DEFAULT_PRICES
        self.warning_threshold = warning_threshold
        self.on_warning = on_warning
        self.bucket_name = bucket_name
        self._spent = 0.0
        self._warned = False
        self._async_lock: asyncio.Lock | None = None

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        """Serialize check -> call -> record across everyone sharing this gate.

        Async routers wrap each budgeted attempt in this hold so that two
        concurrent calls cannot both pass the pre-call check against the
        same remaining budget. The lock lives on the gate — not on any
        router — so the protection also covers several router instances
        sharing one gate. It is created lazily, inside whichever event
        loop first uses it. Sync callers don't need it: their check ->
        call -> record cycle never awaits in between.
        """
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        async with self._async_lock:
            yield

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget_usd - self._spent)

    @property
    def warned(self) -> bool:
        return self._warned

    def check(self, model: str, est_input_tokens: int, est_output_tokens: int) -> float:
        """Return the estimated cost, or raise :class:`BudgetExceeded`."""
        est = estimate_cost(model, est_input_tokens, est_output_tokens, self.prices)
        if self.per_request_cap_usd is not None and est > self.per_request_cap_usd:
            raise BudgetExceeded(
                f"estimated request cost ${est:.4f} exceeds per-request cap "
                f"${self.per_request_cap_usd:.4f} (model {model!r}, bucket {self.bucket_name!r})",
                bucket=self.bucket_name,
            )
        if self._spent + est > self.budget_usd:
            raise BudgetExceeded(
                f"estimated request cost ${est:.4f} would exceed run budget: "
                f"spent ${self._spent:.4f} of ${self.budget_usd:.4f} "
                f"(model {model!r}, bucket {self.bucket_name!r})",
                bucket=self.bucket_name,
            )
        return est

    def record(self, model: str, usage: Usage) -> float:
        """Add the actual cost of a completed call; returns that cost."""
        cost = estimate_cost(model, usage.input_tokens, usage.output_tokens, self.prices)
        self._spent += cost
        if (
            not self._warned
            and self.warning_threshold is not None
            and self._spent >= self.budget_usd * self.warning_threshold
        ):
            self._warned = True
            if self.on_warning is not None:
                self.on_warning(self.bucket_name, self._spent, self.budget_usd)
        return cost


class BudgetManager:
    """Bucketed spend caps: one independent :class:`BudgetGate` per bucket.

    Buckets are arbitrary string keys chosen by the caller (e.g. a tenant ID,
    a task name, or ``"tenant/task"``). Each bucket tracks its own spend,
    cap, warning flag, and async :meth:`BudgetGate.hold` lock — exhausting
    bucket A never blocks bucket B, and concurrent async calls on different
    buckets do not serialize against each other.

    Calls with ``bucket=None`` go to ``default_bucket`` (``"default"``).
    Calls with an explicit bucket name not in ``budgets`` follow
    ``unknown_bucket``:

    - ``"raise"`` (default): raise :class:`KeyError` to catch typos early.
    - ``"create"``: lazily create an independent gate using ``default_budget_usd``.
    - ``"default"``: route spend to ``default_bucket``.
    """

    def __init__(
        self,
        budgets: dict[str, float] | None = None,
        *,
        default_budget_usd: float | None = None,
        per_request_cap_usd: float | None = None,
        per_bucket_request_caps: dict[str, float] | None = None,
        warning_threshold: float | None = 0.8,
        on_warning: WarningCallback | None = None,
        prices: dict[str, tuple[float, float]] | None = None,
        unknown_bucket: UnknownBucketPolicy = "raise",
        default_bucket: str = "default",
    ) -> None:
        if unknown_bucket not in ("raise", "create", "default"):
            raise ValueError(
                f"unknown_bucket must be 'raise', 'create', or 'default', got {unknown_bucket!r}"
            )
        _validate_warning_threshold(warning_threshold)
        if unknown_bucket == "create" and default_budget_usd is None:
            raise ValueError(
                "unknown_bucket='create' requires default_budget_usd to be set"
            )

        self.prices = prices if prices is not None else DEFAULT_PRICES
        self.default_budget_usd = default_budget_usd
        self.per_request_cap_usd = per_request_cap_usd
        self.warning_threshold = warning_threshold
        self.on_warning = on_warning
        self.unknown_bucket: UnknownBucketPolicy = unknown_bucket
        self.default_bucket = default_bucket
        self._per_bucket_caps = dict(per_bucket_request_caps or {})
        self._gates: dict[str, BudgetGate] = {}

        for name, cap in (budgets or {}).items():
            self._gates[name] = self._make_gate(name, cap)

        if default_budget_usd is not None and default_bucket not in self._gates:
            self._gates[default_bucket] = self._make_gate(
                default_bucket, default_budget_usd
            )

        if unknown_bucket == "default" and default_bucket not in self._gates:
            raise ValueError(
                "unknown_bucket='default' requires default_budget_usd or "
                f"{default_bucket!r} in budgets"
            )

    def _make_gate(self, name: str, budget_usd: float) -> BudgetGate:
        req_cap = self._per_bucket_caps.get(name, self.per_request_cap_usd)
        return BudgetGate(
            budget_usd=budget_usd,
            per_request_cap_usd=req_cap,
            prices=self.prices,
            warning_threshold=self.warning_threshold,
            on_warning=self.on_warning,
            bucket_name=name,
        )

    def resolve(self, bucket: str | None = None) -> tuple[str, BudgetGate]:
        """Return ``(resolved_bucket_name, gate)`` for a call."""
        if bucket is None:
            if self.default_bucket not in self._gates:
                raise KeyError(
                    f"no default bucket {self.default_bucket!r} configured; "
                    "pass default_budget_usd or an explicit bucket"
                )
            return self.default_bucket, self._gates[self.default_bucket]

        if bucket in self._gates:
            return bucket, self._gates[bucket]

        if self.unknown_bucket == "raise":
            raise KeyError(f"unknown budget bucket {bucket!r}")
        if self.unknown_bucket == "default":
            return self.default_bucket, self._gates[self.default_bucket]
        # unknown_bucket == "create"
        assert self.default_budget_usd is not None
        gate = self._make_gate(bucket, self.default_budget_usd)
        self._gates[bucket] = gate
        return bucket, gate

    def gate_for(self, bucket: str | None = None) -> BudgetGate:
        """Return the :class:`BudgetGate` backing ``bucket``."""
        _, gate = self.resolve(bucket)
        return gate

    @property
    def buckets(self) -> tuple[str, ...]:
        return tuple(self._gates.keys())

    @property
    def total_spent(self) -> float:
        return sum(g.spent for g in self._gates.values())

    def spent_for(self, bucket: str | None = None) -> float:
        return self.gate_for(bucket).spent

    def remaining_for(self, bucket: str | None = None) -> float:
        return self.gate_for(bucket).remaining

    def warned_for(self, bucket: str | None = None) -> bool:
        return self.gate_for(bucket).warned

    def check(
        self,
        model: str,
        est_input_tokens: int,
        est_output_tokens: int,
        *,
        bucket: str | None = None,
    ) -> float:
        return self.gate_for(bucket).check(model, est_input_tokens, est_output_tokens)

    def record(
        self,
        model: str,
        usage: Usage,
        *,
        bucket: str | None = None,
    ) -> float:
        return self.gate_for(bucket).record(model, usage)

    @asynccontextmanager
    async def hold(self, bucket: str | None = None) -> AsyncIterator[None]:
        gate = self.gate_for(bucket)
        async with gate.hold():
            yield
