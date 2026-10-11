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
