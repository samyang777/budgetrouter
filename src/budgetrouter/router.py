"""Routing rules and fallback across providers and models.

The :class:`Router` picks a (provider, model) pair for each request from an
ordered list of :class:`RouteRule`\\ s — matched on a caller-supplied tag
and/or the estimated input size — falling back to a default route. If the
chosen provider errors, the fallback chain is tried in order. An optional
:class:`~budgetrouter.budget.BudgetGate` is checked before each attempt and
updated after each success; a :class:`BudgetExceeded` error is *not* treated
as a provider failure and propagates immediately.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

from .budget import BudgetGate
from .provider import Completion, Message, Provider, ProviderError


def estimate_tokens(messages: list[Message]) -> int:
    """Rough input-token estimate: ~4 characters per token, minimum 1."""
    chars = sum(len(m.get("content", "")) for m in messages)
    return max(1, math.ceil(chars / 4))


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


class Router:
    def __init__(
        self,
        providers: dict[str, Provider],
        rules: list[RouteRule] | None = None,
        *,
        default: Route,
        fallbacks: list[Route] | None = None,
        budget: BudgetGate | None = None,
        expected_output_tokens: int = 256,
    ) -> None:
        self.providers = providers
        self.rules = rules or []
        self.default = default
        self.fallbacks = fallbacks or []
        self.budget = budget
        self.expected_output_tokens = expected_output_tokens
        referenced = (
            [r.provider for r in self.rules]
            + [default.provider]
            + [f.provider for f in self.fallbacks]
        )
        missing = sorted({p for p in referenced if p not in providers})
        if missing:
            raise ValueError(f"routes reference unknown providers: {missing}")

    def pick_route(self, tag: str | None, input_tokens: int) -> Route:
        for rule in self.rules:
            if rule.matches(tag, input_tokens):
                return rule.route
        return self.default

    def complete(self, messages: list[Message], tag: str | None = None) -> Completion:
        est_input = estimate_tokens(messages)
        primary = self.pick_route(tag, est_input)
        attempts = [primary] + [f for f in self.fallbacks if f != primary]
        errors: list[str] = []
        for route in attempts:
            if self.budget is not None:
                self.budget.check(route.model, est_input, self.expected_output_tokens)
            provider = self.providers[route.provider]
            try:
                completion = provider.complete(messages, route.model)
            except ProviderError as exc:
                errors.append(f"{route.provider}/{route.model}: {exc}")
                continue
            if self.budget is not None:
                self.budget.record(route.model, completion.usage)
            return completion
        raise ProviderError(
            "all routes failed: " + ("; ".join(errors) if errors else "no routes configured")
        )

    async def acomplete(
        self, messages: list[Message], tag: str | None = None
    ) -> Completion:
        """Async twin of :meth:`complete`, with identical semantics.

        Same routing rules, same fallback chain, same budget behavior
        (``BudgetExceeded`` propagates immediately and is never treated as
        a provider failure). Each provider is called through its native
        ``acomplete`` when it defines one; a sync-only provider's
        ``complete`` is run in a worker thread, so existing providers keep
        working unchanged in async applications such as LangGraph.
        """
        est_input = estimate_tokens(messages)
        primary = self.pick_route(tag, est_input)
        attempts = [primary] + [f for f in self.fallbacks if f != primary]
        errors: list[str] = []
        for route in attempts:
            if self.budget is not None:
                self.budget.check(route.model, est_input, self.expected_output_tokens)
            provider = self.providers[route.provider]
            try:
                completion = await self._call_provider_async(
                    provider, messages, route.model
                )
            except ProviderError as exc:
                errors.append(f"{route.provider}/{route.model}: {exc}")
                continue
            if self.budget is not None:
                self.budget.record(route.model, completion.usage)
            return completion
        raise ProviderError(
            "all routes failed: " + ("; ".join(errors) if errors else "no routes configured")
        )

    @staticmethod
    async def _call_provider_async(
        provider: Provider, messages: list[Message], model: str
    ) -> Completion:
        acomplete = getattr(provider, "acomplete", None)
        if acomplete is not None:
            return await acomplete(messages, model)
        return await asyncio.to_thread(provider.complete, messages, model)


class AsyncRouter:
    """Async counterpart of :class:`Router`, with identical semantics.

    Same routing rules (tag and estimated input size, first match wins,
    then the default route), same fallback chain (only
    :class:`ProviderError` falls through), and the same budget behavior:
    the gate is checked before each attempt and updated after each
    success, and :class:`BudgetExceeded` propagates immediately — it is
    never treated as a provider failure, so an exhausted budget does not
    trigger fallback spending.

    The one deliberate difference is concurrency control. When a
    :class:`BudgetGate` is attached, each attempt's check → call →
    record cycle runs under an :class:`asyncio.Lock`. Without that, two
    concurrent calls could both pass the pre-call check against the same
    remaining budget and overshoot the cap; with it, recorded spend can
    never exceed the cap. The trade-off is honesty-worthy: calls through
    one ``AsyncRouter`` serialize *while a budget is attached*. With no
    budget, nothing is locked and calls run fully concurrently.
    """

    def __init__(
        self,
        providers: dict[str, Provider],
        rules: list[RouteRule] | None = None,
        *,
        default: Route,
        fallbacks: list[Route] | None = None,
        budget: BudgetGate | None = None,
        expected_output_tokens: int = 256,
    ) -> None:
        self.providers = providers
        self.rules = rules or []
        self.default = default
        self.fallbacks = fallbacks or []
        self.budget = budget
        self.expected_output_tokens = expected_output_tokens
        self._budget_lock = asyncio.Lock()
        referenced = (
            [r.provider for r in self.rules]
            + [default.provider]
            + [f.provider for f in self.fallbacks]
        )
        missing = sorted({p for p in referenced if p not in providers})
        if missing:
            raise ValueError(f"routes reference unknown providers: {missing}")

    def pick_route(self, tag: str | None, input_tokens: int) -> Route:
        for rule in self.rules:
            if rule.matches(tag, input_tokens):
                return rule.route
        return self.default

    async def complete(
        self, messages: list[Message], tag: str | None = None
    ) -> Completion:
        est_input = estimate_tokens(messages)
        primary = self.pick_route(tag, est_input)
        attempts = [primary] + [f for f in self.fallbacks if f != primary]
        errors: list[str] = []
        for route in attempts:
            provider = self.providers[route.provider]
            try:
                if self.budget is not None:
                    async with self._budget_lock:
                        self.budget.check(
                            route.model, est_input, self.expected_output_tokens
                        )
                        completion = await self._call_provider(
                            provider, messages, route.model
                        )
                        self.budget.record(route.model, completion.usage)
                else:
                    completion = await self._call_provider(
                        provider, messages, route.model
                    )
            except ProviderError as exc:
                errors.append(f"{route.provider}/{route.model}: {exc}")
                continue
            return completion
        raise ProviderError(
            "all routes failed: " + ("; ".join(errors) if errors else "no routes configured")
        )

    @staticmethod
    async def _call_provider(
        provider: Provider, messages: list[Message], model: str
    ) -> Completion:
        complete = getattr(provider, "complete", None)
        if complete is not None and asyncio.iscoroutinefunction(complete):
            return await complete(messages, model)
        acomplete = getattr(provider, "acomplete", None)
        if acomplete is not None:
            return await acomplete(messages, model)
        return await asyncio.to_thread(provider.complete, messages, model)
