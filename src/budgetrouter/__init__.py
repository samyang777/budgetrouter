"""budgetrouter: multi-model routing with budget gates, fallback policies, and usage attribution."""

from .adapters import (
    AnthropicProvider,
    AsyncAnthropicProvider,
    AsyncOpenAICompatibleProvider,
    OpenAICompatibleProvider,
)
from .budget import (
    DEFAULT_PRICES,
    BudgetExceeded,
    BudgetGate,
    BudgetManager,
    estimate_cost,
)
from .eval import (
    CaseResult,
    EvalReport,
    GateResult,
    check_gate,
    load_cases,
    run_eval,
    run_router_eval,
)
from .policy import DEFAULT_FALLBACK_ACTIONS, FallbackPolicy
from .provider import (
    AsyncFakeProvider,
    AsyncProvider,
    AttemptOutcome,
    AttemptRecord,
    Completion,
    FailureKind,
    FakeProvider,
    Provider,
    ProviderError,
    Usage,
)
from .router import AsyncRouter, Route, RouteRule, Router, estimate_tokens
from .usage import GroupSummary, UsageLedger, UsageSummary

__version__ = "0.2.0"

__all__ = [
    "AnthropicProvider",
    "AsyncAnthropicProvider",
    "AsyncFakeProvider",
    "AsyncOpenAICompatibleProvider",
    "AsyncProvider",
    "AsyncRouter",
    "AttemptOutcome",
    "AttemptRecord",
    "BudgetExceeded",
    "BudgetGate",
    "BudgetManager",
    "CaseResult",
    "Completion",
    "DEFAULT_FALLBACK_ACTIONS",
    "DEFAULT_PRICES",
    "EvalReport",
    "FailureKind",
    "FakeProvider",
    "FallbackPolicy",
    "GateResult",
    "GroupSummary",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderError",
    "Route",
    "RouteRule",
    "Router",
    "Usage",
    "UsageLedger",
    "UsageSummary",
    "check_gate",
    "estimate_cost",
    "estimate_tokens",
    "load_cases",
    "run_eval",
    "run_router_eval",
]
