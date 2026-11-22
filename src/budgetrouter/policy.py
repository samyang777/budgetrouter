"""Configurable fallback policies keyed by provider failure kind.

When a provider attempt raises :class:`~budgetrouter.provider.ProviderError`,
the router consults a :class:`FallbackPolicy` to decide what happens next:

- ``"fallback"`` — move on to the next candidate route in the fallback chain.
- ``"retry_then_fallback"`` — retry the same route up to ``max_retries``
  times, then fall back if it still fails.
- ``"abort"`` — stop immediately and propagate the error without trying any
  further routes (the default for ``"auth"`` failures, since retrying a bad
  key masks configuration bugs).
"""

from __future__ import annotations

from collections.abc import Mapping

from .provider import FailureKind

FallbackAction = str  # Literal["fallback", "retry_then_fallback", "abort"]

VALID_ACTIONS: frozenset[str] = frozenset(
    {"fallback", "retry_then_fallback", "abort"}
)

DEFAULT_FALLBACK_ACTIONS: dict[FailureKind, str] = {
    "timeout": "fallback",
    "rate_limit": "fallback",
    "server_error": "fallback",
    "auth": "abort",
    "invalid_request": "fallback",
    "transport": "fallback",
    "bad_response": "fallback",
    "unknown": "fallback",
}


class FallbackPolicy:
    """Policy deciding how the router reacts to each :class:`FailureKind`.

    Unspecified failure kinds inherit from :data:`DEFAULT_FALLBACK_ACTIONS`.
    When ``skip_same_provider_on_auth`` is ``True`` and ``auth`` maps to
    ``"fallback"``, remaining fallback routes sharing the failed provider are
    skipped while routes on *other* providers are still tried.
    """

    def __init__(
        self,
        actions: Mapping[str, str] | None = None,
        *,
        max_retries: int = 1,
        skip_same_provider_on_auth: bool = False,
    ) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries cannot be negative: {max_retries}")
        merged: dict[str, str] = dict(DEFAULT_FALLBACK_ACTIONS)
        if actions:
            for kind, action in actions.items():
                if action not in VALID_ACTIONS:
                    raise ValueError(
                        f"unknown fallback action {action!r} for kind {kind!r}; "
                        f"choose from {sorted(VALID_ACTIONS)}"
                    )
                merged[kind] = action
        self.actions: dict[str, str] = merged
        self.max_retries = max_retries
        self.skip_same_provider_on_auth = skip_same_provider_on_auth

    def action_for(self, kind: str) -> str:
        """Return the configured action for ``kind`` (defaults to ``"fallback"``)."""
        return self.actions.get(kind, "fallback")
