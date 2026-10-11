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
import os
import urllib.error
import urllib.request

from .provider import Completion, Message, ProviderError, Usage

ANTHROPIC_VERSION = "2023-06-01"
"""Value sent in the ``anthropic-version`` header."""


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
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
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
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"{provider_name}: undecodable response from {url}: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise ProviderError(f"{provider_name}: unexpected response shape from {url}")
    return decoded


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
            usage_data = data.get("usage") or {}
            usage = Usage(
                input_tokens=int(usage_data.get("prompt_tokens", 0)),
                output_tokens=int(usage_data.get("completion_tokens", 0)),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"{self.name}: malformed chat-completions response: {exc}"
            ) from exc
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
            usage_data = data.get("usage") or {}
            usage = Usage(
                input_tokens=int(usage_data.get("input_tokens", 0)),
                output_tokens=int(usage_data.get("output_tokens", 0)),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ProviderError(
                f"{self.name}: malformed messages response: {exc}"
            ) from exc
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
