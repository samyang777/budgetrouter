"""Real provider adapters, built on the standard library only.

These adapters speak plain HTTPS via :mod:`urllib.request`, so the package
keeps its zero-runtime-dependency promise — no vendor SDKs are required.
Both implement the :class:`~budgetrouter.provider.Provider` protocol and can
be pointed at any base URL, which is also how the test-suite runs them
against a local stub server.

API keys are never hard-coded: pass ``api_key`` explicitly or name an
environment variable with ``api_key_env`` (defaults to the vendor's usual
variable) and the key is read from the environment at construction time.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import urllib.error
import urllib.request

from .provider import Completion, Message, ProviderError, Usage

ANTHROPIC_VERSION = "2023-06-01"
"""Value sent in the ``anthropic-version`` header."""

USER_AGENT = "budgetrouter/0.1.0"
"""Value sent in the ``User-Agent`` header of every adapter request.

urllib's default ``Python-urllib/x.y`` agent is rejected outright by
some API gateways (Cloudflare-fronted endpoints answer HTTP 403, error
1010), so the adapters identify themselves explicitly instead.
"""


def _resolve_api_key(
    api_key: str | None, api_key_env: str, provider_name: str
) -> str:
    if api_key:
        return api_key
    value = os.environ.get(api_key_env)
    if not value:
        raise ProviderError(
            f"{provider_name}: no API key given and environment variable "
            f"{api_key_env} is not set"
        )
    return value


def _post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: float,
    provider_name: str,
) -> dict:
    """POST ``payload`` as JSON and return the decoded response body.

    Any transport failure, non-2xx status, or undecodable body becomes a
    :class:`ProviderError`, so the router's fallback chain can react.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            **headers,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_bytes = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise ProviderError(
            f"{provider_name}: HTTP {exc.code} from {url}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderError(
            f"{provider_name}: request to {url} failed: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise ProviderError(f"{provider_name}: request to {url} timed out") from exc
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProviderError(
            f"{provider_name}: undecodable response from {url}: {exc}"
        ) from exc
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"{provider_name}: undecodable response from {url}: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise ProviderError(f"{provider_name}: unexpected response shape from {url}")
    return decoded


def _estimate_tokens(text: str) -> int:
    """Rough token estimate for a text: ~4 characters per token."""
    return math.ceil(len(text) / 4) if text else 0


def _parse_usage(
    usage_data: object,
    *,
    input_key: str,
    output_key: str,
    input_fallback: int,
    output_fallback: int,
    provider_name: str,
) -> Usage:
    """Build a :class:`Usage` from a provider's usage block.

    A provider that omits usage — or just one of its fields — must not
    make the call free: the missing counts are estimated from the
    request and response text (the ``input_fallback`` / ``output_fallback``
    arguments), using the heuristic above. Values that are present but
    negative or non-numeric are malformed and raise :class:`ProviderError`;
    a negative count would otherwise *reduce* recorded spend in the
    budget gate.
    """
    if usage_data is None:
        return Usage(input_tokens=input_fallback, output_tokens=output_fallback)
    if not isinstance(usage_data, dict):
        raise ProviderError(
            f"{provider_name}: malformed usage block: {usage_data!r}"
        )
    parsed: dict[str, int] = {}
    for key, fallback in (
        (input_key, input_fallback),
        (output_key, output_fallback),
    ):
        raw_value = usage_data.get(key)
        if raw_value is None:
            parsed[key] = fallback
            continue
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ProviderError(
                f"{provider_name}: malformed usage value {key}={raw_value!r}"
            ) from exc
        if value < 0:
            raise ProviderError(
                f"{provider_name}: negative usage value {key}={value}"
            )
        parsed[key] = value
    return Usage(
        input_tokens=parsed[input_key], output_tokens=parsed[output_key]
    )


class OpenAICompatibleProvider:
    """Provider for OpenAI's chat-completions API and compatible servers.

    Works with OpenAI itself and with any server exposing the same
    ``POST {base_url}/chat/completions`` shape (many gateways and local
    model servers do). ``model`` is the default model, used when a call
    does not name one; the router passes the routed model on each call.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 30.0,
        name: str = "openai",
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._api_key = _resolve_api_key(api_key, api_key_env, name)

    def complete(self, messages: list[Message], model: str) -> Completion:
        use_model = model or self.model
        data = _post_json(
            f"{self.base_url}/chat/completions",
            {"model": use_model, "messages": messages},
            {"Authorization": f"Bearer {self._api_key}"},
            self.timeout,
            self.name,
        )
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"{self.name}: malformed chat-completions response: {exc}"
            ) from exc
        usage = _parse_usage(
            data.get("usage"),
            input_key="prompt_tokens",
            output_key="completion_tokens",
            input_fallback=_estimate_tokens(
                "".join(m.get("content", "") for m in messages)
            ),
            output_fallback=_estimate_tokens(text),
            provider_name=self.name,
        )
        return Completion(
            text=text, model=use_model, provider=self.name, usage=usage
        )

    async def acomplete(self, messages: list[Message], model: str) -> Completion:
        """Async twin of :meth:`complete`.

        The HTTP call itself is the blocking urllib request, run in a
        worker thread so the event loop stays free; request/response
        handling is shared with the sync path, byte for byte.
        """
        return await asyncio.to_thread(self.complete, messages, model)


class AnthropicProvider:
    """Provider for Anthropic's messages API (``POST /v1/messages``).

    Anthropic takes the system prompt outside the message list, so any
    ``role == "system"`` messages are split out and sent as the top-level
    ``system`` field. ``max_tokens`` is required by the API and defaults to
    1024.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        timeout: float = 30.0,
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 1024,
        name: str = "anthropic",
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._api_key = _resolve_api_key(api_key, api_key_env, name)

    def complete(self, messages: list[Message], model: str) -> Completion:
        use_model = model or self.model
        system = "\n\n".join(
            m.get("content", "") for m in messages if m.get("role") == "system"
        )
        conversation = [m for m in messages if m.get("role") != "system"]
        payload: dict = {
            "model": use_model,
            "messages": conversation,
            "max_tokens": self.max_tokens,
        }
        if system:
            payload["system"] = system
        data = _post_json(
            f"{self.base_url}/v1/messages",
            payload,
            {
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            self.timeout,
            self.name,
        )
        try:
            text = "".join(
                block.get("text", "")
                for block in data["content"]
                if block.get("type") == "text"
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ProviderError(
                f"{self.name}: malformed messages response: {exc}"
            ) from exc
        usage = _parse_usage(
            data.get("usage"),
            input_key="input_tokens",
            output_key="output_tokens",
            input_fallback=_estimate_tokens(
                "".join(m.get("content", "") for m in messages)
            ),
            output_fallback=_estimate_tokens(text),
            provider_name=self.name,
        )
        return Completion(
            text=text, model=use_model, provider=self.name, usage=usage
        )

    async def acomplete(self, messages: list[Message], model: str) -> Completion:
        """Async twin of :meth:`complete`; see the OpenAI adapter's note."""
        return await asyncio.to_thread(self.complete, messages, model)


class AsyncOpenAICompatibleProvider:
    """Async counterpart of :class:`OpenAICompatibleProvider`.

    Same constructor, same wire format, same parsing — the request is
    built and answered by the sync implementation. Because the package
    promises zero runtime dependencies, there is no async HTTP client
    underneath: :mod:`urllib` is blocking, so the call runs via
    :func:`asyncio.to_thread`. That is honestly scoped — the event loop
    is never blocked and concurrent calls genuinely overlap, at the cost
    of one worker thread per in-flight request.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 30.0,
        name: str = "openai",
    ) -> None:
        self._sync = OpenAICompatibleProvider(
            base_url,
            model,
            api_key=api_key,
            api_key_env=api_key_env,
            timeout=timeout,
            name=name,
        )
        self.name = self._sync.name
        self.base_url = self._sync.base_url
        self.model = self._sync.model
        self.timeout = self._sync.timeout

    async def complete(self, messages: list[Message], model: str) -> Completion:
        return await asyncio.to_thread(self._sync.complete, messages, model)


class AsyncAnthropicProvider:
    """Async counterpart of :class:`AnthropicProvider`.

    Same constructor and behavior as the sync adapter (including the
    system-message split), with the blocking urllib call offloaded via
    :func:`asyncio.to_thread` — see :class:`AsyncOpenAICompatibleProvider`
    for why, and what that does and does not promise.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        timeout: float = 30.0,
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 1024,
        name: str = "anthropic",
    ) -> None:
        self._sync = AnthropicProvider(
            model,
            api_key=api_key,
            api_key_env=api_key_env,
            timeout=timeout,
            base_url=base_url,
            max_tokens=max_tokens,
            name=name,
        )
        self.name = self._sync.name
        self.base_url = self._sync.base_url
        self.model = self._sync.model
        self.timeout = self._sync.timeout
        self.max_tokens = self._sync.max_tokens

    async def complete(self, messages: list[Message], model: str) -> Completion:
        return await asyncio.to_thread(self._sync.complete, messages, model)
