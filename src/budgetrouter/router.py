"""Routing rules, failure-aware fallback, budget gating, and usage recording.

The :class:`Router` and :class:`AsyncRouter` pick a ``(provider, model)`` pair
for each request from an ordered list of :class:`RouteRule`\\ s — matched on a
caller-supplied tag and/or the estimated input size — falling back to a default
route. When a provider raises :class:`~budgetrouter.provider.ProviderError`,
the router consults its :class:`~budgetrouter.policy.FallbackPolicy` to abort,
retry, or move down the fallback chain.

Every attempt (success, failure, or budget rejection) is recorded as an
:class:`~budgetrouter.provider.AttemptRecord`, attached to the returned
:class:`~budgetrouter.provider.Completion` or raised exception, and appended
to the router's :class:`~budgetrouter.usage.UsageLedger`.

An optional :class:`~budgetrouter.budget.BudgetGate` or
:class:`~budgetrouter.budget.BudgetManager` is checked before each attempt and
updated after each success; :class:`~budgetrouter.budget.BudgetExceeded` is
*never* treated as a provider failure and propagates immediately.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

from .budget import DEFAULT_PRICES, BudgetExceeded, BudgetGate, BudgetManager, estimate_cost
from .policy import FallbackPolicy
from .provider import (
    AsyncProvider,
    AttemptRecord,
    Completion,
    FailureKind,
    Message,
    Provider,
    ProviderError,
)
from .usage import UsageLedger, UsageSummary


def estimate_tokens(messages: list[Message]) -> int:
    """Rough input-token estimate: ~4 characters per token, minimum 1."""
    chars = sum(len(m.get("content", "")) for m in messages)
    return max(1, math.ceil(chars / 4))


def _safe_estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    prices: dict[str, tuple[float, float]],
) -> float | None:
    if model not in prices:
        return None
    return estimate_cost(model, input_tokens, output_tokens, prices)


@dataclass(frozen=True)
class Route:
    provider: str
    model: str


@dataclass(frozen=True)
class RouteRule:
    """One routing rule; the first matching rule wins.

    Matches when ``tag`` is None or equals the request's tag, and the
    estimated input size is within [min_input_tokens, max_input_tokens].
    """

    provider: str
    model: str
    tag: str | None = None
    min_input_tokens: int = 0
    max_input_tokens: int | None = None

    def matches(self, tag: str | None, input_tokens: int) -> bool:
        if self.tag is not None and self.tag != tag:
            return False
        if input_tokens < self.min_input_tokens:
            return False
        if self.max_input_tokens is not None and input_tokens > self.max_input_tokens:
            return False
        return True

    @property
    def route(self) -> Route:
        return Route(provider=self.provider, model=self.model)


class _BaseRouter:
    """Shared configuration, route selection, and async execution core."""

    def __init__(
        self,
        providers: Mapping[str, Provider | AsyncProvider],
        rules: list[RouteRule] | None = None,
        *,
        default: Route,
        fallbacks: list[Route] | None = None,
        budget: BudgetGate | BudgetManager | None = None,
        expected_output_tokens: int = 256,
        fallback_policy: FallbackPolicy | None = None,
        usage: UsageLedger | None = None,
        prices: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.providers = dict(providers)
        self.rules = rules or []
        self.default = default
        self.fallbacks = fallbacks or []
        self.budget = budget
        self.expected_output_tokens = expected_output_tokens
        self.fallback_policy = fallback_policy or FallbackPolicy()
        self.usage = usage if usage is not None else UsageLedger()
        self._prices = prices if prices is not None else DEFAULT_PRICES
        referenced = (
            [r.provider for r in self.rules]
            + [default.provider]
            + [f.provider for f in self.fallbacks]
        )
        missing = sorted({p for p in referenced if p not in self.providers})
        if missing:
            raise ValueError(f"routes reference unknown providers: {missing}")

    def pick_route(self, tag: str | None, input_tokens: int) -> Route:
        for rule in self.rules:
            if rule.matches(tag, input_tokens):
                return rule.route
        return self.default

    def usage_summary(
        self, group_by: str | tuple[str, ...] = ("bucket", "model")
    ) -> UsageSummary:
        """Return aggregated usage from :attr:`usage`."""
        return self.usage.summary(group_by=group_by)

    def _resolve_budget(
        self, bucket: str | None
    ) -> tuple[str, BudgetGate | None, dict[str, tuple[float, float]]]:
        if isinstance(self.budget, BudgetManager):
            bucket_name, gate = self.budget.resolve(bucket)
            return bucket_name, gate, gate.prices
        if isinstance(self.budget, BudgetGate):
            return bucket or self.budget.bucket_name, self.budget, self.budget.prices
        return bucket or "default", None, self._prices

    def _plan_routes(
        self, messages: list[Message], tag: str | None
    ) -> tuple[int, Route, list[Route]]:
        est_input = estimate_tokens(messages)
        primary = self.pick_route(tag, est_input)
        attempts = [primary] + [f for f in self.fallbacks if f != primary]
        return est_input, primary, attempts

    def _record_attempt(
        self, records: list[AttemptRecord], attempt: AttemptRecord
    ) -> None:
        records.append(attempt)
        self.usage.record(attempt)

    @staticmethod
    def _build_exhausted_error(
        errors: list[str],
        kinds: list[FailureKind],
        records: list[AttemptRecord],
    ) -> ProviderError:
        unique_kinds = set(kinds)
        summary_kind: FailureKind = (
            kinds[0] if len(unique_kinds) == 1 else "unknown"
        )
        detail = "; ".join(errors) if errors else "no routes configured"
        return ProviderError(
            f"all routes failed: {detail}",
            kind=summary_kind,
            attempts=tuple(records),
        )

    @staticmethod
    async def _call_provider_async(
        provider: Provider | AsyncProvider, messages: list[Message], model: str
    ) -> Completion:
        complete = getattr(provider, "complete", None)
        if complete is not None and asyncio.iscoroutinefunction(complete):
            return await complete(messages, model)
        acomplete = getattr(provider, "acomplete", None)
        if acomplete is not None:
            return await acomplete(messages, model)
        return await asyncio.to_thread(provider.complete, messages, model)

    async def _acomplete_impl(
        self,
        messages: list[Message],
        tag: str | None = None,
        *,
        bucket: str | None = None,
    ) -> Completion:
        bucket_name, gate, price_table = self._resolve_budget(bucket)
        est_input, primary, candidate_routes = self._plan_routes(messages, tag)
        records: list[AttemptRecord] = []
        errors: list[str] = []
        kinds: list[FailureKind] = []
        skipped_providers: set[str] = set()

        for route in candidate_routes:
            if route.provider in skipped_providers:
                continue
            provider = self.providers[route.provider]
            is_primary = route == primary
            retries_done = 0

            while True:
                est_cost = _safe_estimate_cost(
                    route.model, est_input, self.expected_output_tokens, price_table
                )
                start = time.perf_counter()
                try:
                    if gate is not None:
                        async with gate.hold():
                            est_cost = gate.check(
                                route.model, est_input, self.expected_output_tokens
                            )
                            completion = await self._call_provider_async(
                                provider, messages, route.model
                            )
                            actual_cost: float | None = gate.record(
                                route.model, completion.usage
                            )
                    else:
                        completion = await self._call_provider_async(
                            provider, messages, route.model
                        )
                        actual_cost = _safe_estimate_cost(
                            route.model,
                            completion.usage.input_tokens,
                            completion.usage.output_tokens,
                            price_table,
                        )
                except BudgetExceeded as exc:
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    self._record_attempt(
                        records,
                        AttemptRecord(
                            provider=route.provider,
                            model=route.model,
                            outcome="budget_exceeded",
                            latency_ms=latency_ms,
                            is_primary=is_primary,
                            bucket=bucket_name,
                            error_message=str(exc),
                            estimated_cost_usd=est_cost,
                            actual_cost_usd=0.0,
                        ),
                    )
                    exc.attempts = tuple(records)
                    if exc.bucket is None:
                        exc.bucket = bucket_name
                    raise
                except ProviderError as exc:
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    self._record_attempt(
                        records,
                        AttemptRecord(
                            provider=route.provider,
                            model=route.model,
                            outcome=exc.kind,
                            latency_ms=latency_ms,
                            is_primary=is_primary,
                            bucket=bucket_name,
                            status_code=exc.status_code,
                            error_message=str(exc),
                            estimated_cost_usd=est_cost,
                            actual_cost_usd=0.0,
                        ),
                    )
                    errors.append(
                        f"{route.provider}/{route.model} [{exc.kind}]: {exc}"
                    )
                    kinds.append(exc.kind)
                    action = self.fallback_policy.action_for(exc.kind)
                    if action == "abort":
                        raise ProviderError(
                            str(exc),
                            kind=exc.kind,
                            status_code=exc.status_code,
                            attempts=tuple(records),
                        ) from exc
                    if (
                        action == "retry_then_fallback"
                        and retries_done < self.fallback_policy.max_retries
                    ):
                        retries_done += 1
                        continue
                    if (
                        exc.kind == "auth"
                        and self.fallback_policy.skip_same_provider_on_auth
                    ):
                        skipped_providers.add(route.provider)
                    break

                latency_ms = (time.perf_counter() - start) * 1000.0
                self._record_attempt(
                    records,
                    AttemptRecord(
                        provider=route.provider,
                        model=route.model,
                        outcome="success",
                        latency_ms=latency_ms,
                        is_primary=is_primary,
                        bucket=bucket_name,
                        input_tokens=completion.usage.input_tokens,
                        output_tokens=completion.usage.output_tokens,
                        estimated_cost_usd=est_cost,
                        actual_cost_usd=actual_cost,
                    ),
                )
                return Completion(
                    text=completion.text,
                    model=completion.model,
                    provider=completion.provider,
                    usage=completion.usage,
                    attempts=tuple(records),
                )

        raise self._build_exhausted_error(errors, kinds, records)


class Router(_BaseRouter):
    """Sync router with an async twin (:meth:`acomplete`)."""

    def complete(
        self,
        messages: list[Message],
        tag: str | None = None,
        *,
        bucket: str | None = None,
    ) -> Completion:
        bucket_name, gate, price_table = self._resolve_budget(bucket)
        est_input, primary, candidate_routes = self._plan_routes(messages, tag)
        records: list[AttemptRecord] = []
        errors: list[str] = []
        kinds: list[FailureKind] = []
        skipped_providers: set[str] = set()

        for route in candidate_routes:
            if route.provider in skipped_providers:
                continue
            provider = self.providers[route.provider]
            is_primary = route == primary
            retries_done = 0

            while True:
                est_cost = _safe_estimate_cost(
                    route.model, est_input, self.expected_output_tokens, price_table
                )
                start = time.perf_counter()
                try:
                    if gate is not None:
                        est_cost = gate.check(
                            route.model, est_input, self.expected_output_tokens
                        )
                    completion = provider.complete(messages, route.model)  # type: ignore[union-attr]
                    if gate is not None:
                        actual_cost: float | None = gate.record(
                            route.model, completion.usage
                        )
                    else:
                        actual_cost = _safe_estimate_cost(
                            route.model,
                            completion.usage.input_tokens,
                            completion.usage.output_tokens,
                            price_table,
                        )
                except BudgetExceeded as exc:
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    self._record_attempt(
                        records,
                        AttemptRecord(
                            provider=route.provider,
                            model=route.model,
                            outcome="budget_exceeded",
                            latency_ms=latency_ms,
                            is_primary=is_primary,
                            bucket=bucket_name,
                            error_message=str(exc),
                            estimated_cost_usd=est_cost,
                            actual_cost_usd=0.0,
                        ),
                    )
                    exc.attempts = tuple(records)
                    if exc.bucket is None:
                        exc.bucket = bucket_name
                    raise
                except ProviderError as exc:
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    self._record_attempt(
                        records,
                        AttemptRecord(
                            provider=route.provider,
                            model=route.model,
                            outcome=exc.kind,
                            latency_ms=latency_ms,
                            is_primary=is_primary,
                            bucket=bucket_name,
                            status_code=exc.status_code,
                            error_message=str(exc),
                            estimated_cost_usd=est_cost,
                            actual_cost_usd=0.0,
                        ),
                    )
                    errors.append(
                        f"{route.provider}/{route.model} [{exc.kind}]: {exc}"
                    )
                    kinds.append(exc.kind)
                    action = self.fallback_policy.action_for(exc.kind)
                    if action == "abort":
                        raise ProviderError(
                            str(exc),
                            kind=exc.kind,
                            status_code=exc.status_code,
                            attempts=tuple(records),
                        ) from exc
                    if (
                        action == "retry_then_fallback"
                        and retries_done < self.fallback_policy.max_retries
                    ):
                        retries_done += 1
                        continue
                    if (
                        exc.kind == "auth"
                        and self.fallback_policy.skip_same_provider_on_auth
                    ):
                        skipped_providers.add(route.provider)
                    break

                latency_ms = (time.perf_counter() - start) * 1000.0
                self._record_attempt(
                    records,
                    AttemptRecord(
                        provider=route.provider,
                        model=route.model,
                        outcome="success",
                        latency_ms=latency_ms,
                        is_primary=is_primary,
                        bucket=bucket_name,
                        input_tokens=completion.usage.input_tokens,
                        output_tokens=completion.usage.output_tokens,
                        estimated_cost_usd=est_cost,
                        actual_cost_usd=actual_cost,
                    ),
                )
                return Completion(
                    text=completion.text,
                    model=completion.model,
                    provider=completion.provider,
                    usage=completion.usage,
                    attempts=tuple(records),
                )

        raise self._build_exhausted_error(errors, kinds, records)

    async def acomplete(
        self,
        messages: list[Message],
        tag: str | None = None,
        *,
        bucket: str | None = None,
    ) -> Completion:
        """Async twin of :meth:`complete`, with identical semantics.

        Same routing rules, same fallback policy, same attempt recording, and
        same budget behavior (``BudgetExceeded`` propagates immediately and is
        never treated as a provider failure). When a budget is attached, each
        attempt's check -> call -> record cycle runs under the gate's lock
        (:meth:`BudgetGate.hold`).
        """
        return await self._acomplete_impl(messages, tag=tag, bucket=bucket)


class AsyncRouter(_BaseRouter):
    """Async counterpart of :class:`Router`, with identical semantics.

    Accepts async providers (:class:`AsyncProvider`), providers defining
    ``acomplete``, and sync-only providers (bridged via
    :func:`asyncio.to_thread`). When a :class:`BudgetGate` or
    :class:`BudgetManager` is attached, each attempt's check -> call -> record
    cycle runs under the target gate's :meth:`BudgetGate.hold` lock.
    """

    async def complete(
        self,
        messages: list[Message],
        tag: str | None = None,
        *,
        bucket: str | None = None,
    ) -> Completion:
        return await self._acomplete_impl(messages, tag=tag, bucket=bucket)
