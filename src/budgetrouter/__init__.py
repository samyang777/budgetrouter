"""budgetrouter: multi-model routing with budget gates and fallback."""

from .adapters import (
    AnthropicProvider,
    AsyncAnthropicProvider,
    AsyncOpenAICompatibleProvider,
    OpenAICompatibleProvider,
)
from .budget import DEFAULT_PRICES, BudgetExceeded, BudgetGate, estimate_cost
from .eval import EvalReport, load_cases, run_eval
from .provider import (
    AsyncFakeProvider,
    AsyncProvider,
    Completion,
    FakeProvider,
    Provider,
    ProviderError,
    Usage,
)
from .router import AsyncRouter, Route, RouteRule, Router, estimate_tokens

__version__ = "0.1.0"

__all__ = [
    "AnthropicProvider",
    "AsyncAnthropicProvider",
    "AsyncFakeProvider",
    "AsyncOpenAICompatibleProvider",
    "AsyncProvider",
    "AsyncRouter",
    "BudgetExceeded",
    "BudgetGate",
    "Completion",
    "DEFAULT_PRICES",
    "EvalReport",
    "FakeProvider",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderError",
    "Route",
    "RouteRule",
    "Router",
    "Usage",
    "estimate_cost",
    "estimate_tokens",
    "load_cases",
    "run_eval",
]
