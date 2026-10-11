"""Budget gates: cumulative cost tracking with run and per-request caps.

Costs are estimated from a simple editable price table of USD per 1K tokens
(input, output) keyed by model name. The gate is consulted *before* a call
(with estimated token counts) and updated *after* it (with actual usage), so
a runaway agent loop hits a clear :class:`BudgetExceeded` error instead of a
surprise bill.
"""

from __future__ import annotations

from .provider import Usage

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


class BudgetGate:
    """Track cumulative spend and refuse calls that would exceed the caps.

    ``budget_usd`` caps the whole run; ``per_request_cap_usd`` (optional)
    caps any single call, which catches accidental calls to a very expensive
    model even when the run budget is still healthy.
    """

    def __init__(
        self,
        budget_usd: float,
        *,
        per_request_cap_usd: float | None = None,
        prices: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        self.budget_usd = budget_usd
        self.per_request_cap_usd = per_request_cap_usd
        self.prices = prices if prices is not None else DEFAULT_PRICES
        self._spent = 0.0

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget_usd - self._spent)

    def check(self, model: str, est_input_tokens: int, est_output_tokens: int) -> float:
        """Return the estimated cost, or raise :class:`BudgetExceeded`."""
        est = estimate_cost(model, est_input_tokens, est_output_tokens, self.prices)
        if self.per_request_cap_usd is not None and est > self.per_request_cap_usd:
            raise BudgetExceeded(
                f"estimated request cost ${est:.4f} exceeds per-request cap "
                f"${self.per_request_cap_usd:.4f} (model {model!r})"
            )
        if self._spent + est > self.budget_usd:
            raise BudgetExceeded(
                f"estimated request cost ${est:.4f} would exceed run budget: "
                f"spent ${self._spent:.4f} of ${self.budget_usd:.4f} (model {model!r})"
            )
        return est

    def record(self, model: str, usage: Usage) -> float:
        """Add the actual cost of a completed call; returns that cost."""
        cost = estimate_cost(model, usage.input_tokens, usage.output_tokens, self.prices)
        self._spent += cost
        return cost
