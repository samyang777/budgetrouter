"""Provider abstraction: a minimal protocol any model backend can satisfy.

A provider takes a list of chat messages and a model name, and returns a
:class:`Completion` with the generated text and token usage. Real providers
(HTTP APIs, local models, ...) implement this protocol; :class:`FakeProvider`
is a deterministic in-memory implementation for tests and demos.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Message = dict[str, str]
"""A single chat message, e.g. ``{"role": "user", "content": "hi"}``."""

FailureKind = Literal[
    "timeout",
    "rate_limit",
    "server_error",
    "auth",
    "invalid_request",
    "transport",
    "bad_response",
    "unknown",
]
"""Classification of a :class:`ProviderError`."""

AttemptOutcome = Literal[
    "success",
    "timeout",
    "rate_limit",
    "server_error",
    "auth",
    "invalid_request",
    "transport",
    "bad_response",
    "budget_exceeded",
    "unknown",
]
"""Outcome recorded on an :class:`AttemptRecord`."""


@dataclass(frozen=True)
class AttemptRecord:
    """Structured record of one routed attempt (success, failure, or budget block)."""

    provider: str
    model: str
    outcome: AttemptOutcome
    latency_ms: float
    is_primary: bool
    bucket: str | None = None
    status_code: int | None = None
    error_message: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
    actual_cost_usd: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.outcome == "success"

    @property
    def downgraded(self) -> bool:
        return self.succeeded and not self.is_primary

    @property
    def route_label(self) -> str:
        return f"{self.provider}/{self.model}"


class ProviderError(RuntimeError):
    """Raised when a provider call fails (the router may fall back)."""

    def __init__(
        self,
        message: str = "",
        *,
        kind: FailureKind = "unknown",
        status_code: int | None = None,
        attempts: tuple[AttemptRecord, ...] = (),
    ) -> None:
        super().__init__(message)
        self.kind: FailureKind = kind
        self.status_code: int | None = status_code
        self.attempts: tuple[AttemptRecord, ...] = tuple(attempts)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        # A negative count is never legitimate usage; left through, it
        # would *reduce* recorded spend in the budget gate.
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError(
                "token counts cannot be negative: "
                f"input={self.input_tokens}, output={self.output_tokens}"
            )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Completion:
    text: str
    model: str
    provider: str
    usage: Usage
    attempts: tuple[AttemptRecord, ...] = ()


@runtime_checkable
class Provider(Protocol):
    """Anything with a ``name`` and a ``complete`` method is a provider."""

    name: str

    def complete(self, messages: list[Message], model: str) -> Completion:
        ...


@runtime_checkable
class AsyncProvider(Protocol):
    """Async counterpart of :class:`Provider`, for :class:`AsyncRouter`.

    Same contract as :class:`Provider`, but ``complete`` is a coroutine.
    ``AsyncRouter`` also accepts providers that only define an ``acomplete``
    method (the style used by the bundled sync adapters) and plain sync
    providers, which it bridges through a worker thread.
    """

    name: str

    async def complete(self, messages: list[Message], model: str) -> Completion:
        ...


class FakeProvider:
    """Deterministic in-memory provider for tests and demos.

    ``responses`` may be a single string (returned for every model), a dict
    keyed by model name (with ``"*"`` as a wildcard), or ``None`` to echo the
    last message back. ``fail_times`` makes the first N calls raise
    :class:`ProviderError`, which is how fallback behavior is exercised.
    Token counts in the returned usage are fixed and configurable.
    """

    def __init__(
        self,
        responses: dict[str, str] | str | None = None,
        *,
        name: str = "fake",
        input_tokens: int = 12,
        output_tokens: int = 8,
        fail_times: int = 0,
        fail_kind: FailureKind = "unknown",
        fail_status_code: int | None = None,
    ) -> None:
        self.name = name
        self._responses = responses
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._fail_times = fail_times
        self._fail_kind: FailureKind = fail_kind
        self._fail_status_code = fail_status_code
        self.calls: list[tuple[str, list[Message]]] = []

    def complete(self, messages: list[Message], model: str) -> Completion:
        self.calls.append((model, messages))
        if self._fail_times > 0:
            self._fail_times -= 1
            raise ProviderError(
                f"{self.name}: simulated failure for model {model!r}",
                kind=self._fail_kind,
                status_code=self._fail_status_code,
            )
        text = self._pick_response(model, messages)
        return Completion(
            text=text,
            model=model,
            provider=self.name,
            usage=Usage(input_tokens=self._input_tokens, output_tokens=self._output_tokens),
        )

    async def acomplete(self, messages: list[Message], model: str) -> Completion:
        """Async twin of :meth:`complete`; in-memory, so no I/O to await."""
        return self.complete(messages, model)

    def _pick_response(self, model: str, messages: list[Message]) -> str:
        if isinstance(self._responses, dict):
            if model in self._responses:
                return self._responses[model]
            if "*" in self._responses:
                return self._responses["*"]
        elif isinstance(self._responses, str):
            return self._responses
        last = messages[-1].get("content", "") if messages else ""
        return f"echo: {last}"


class AsyncFakeProvider(FakeProvider):
    """Async twin of :class:`FakeProvider`, for :class:`AsyncRouter`.

    Behavior mirrors the sync fake exactly — same canned responses keyed
    by model (``"*"`` wildcard, string, or echo), same fixed token counts,
    same ``fail_times`` simulated failures, same ``calls`` log — because
    the answering logic *is* the sync implementation. The only addition is
    ``delay``: an optional artificial latency (seconds) awaited before
    answering, which makes concurrency behavior observable in tests and
    demos.
    """

    def __init__(
        self,
        responses: dict[str, str] | str | None = None,
        *,
        name: str = "fake",
        input_tokens: int = 12,
        output_tokens: int = 8,
        fail_times: int = 0,
        fail_kind: FailureKind = "unknown",
        fail_status_code: int | None = None,
        delay: float = 0.0,
    ) -> None:
        super().__init__(
            responses,
            name=name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            fail_times=fail_times,
            fail_kind=fail_kind,
            fail_status_code=fail_status_code,
        )
        self.delay = delay

    async def complete(self, messages: list[Message], model: str) -> Completion:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return super().complete(messages, model)

    async def acomplete(self, messages: list[Message], model: str) -> Completion:
        """Keep the inherited ``acomplete`` entry point consistent.

        ``FakeProvider.acomplete`` dispatches to ``self.complete``, which
        is a coroutine on this class — so it is overridden here to await
        it properly instead of returning the coroutine itself.
        """
        return await self.complete(messages, model)
