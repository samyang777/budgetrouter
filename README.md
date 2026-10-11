# budgetrouter

Lightweight multi-model routing for LLM and agent applications: pick the
right model for each request, fall back when a provider fails, and stop the
run when the budget runs out — in a few hundred lines, with no dependencies.

## Why

Agent workloads make a lot of model calls, and two failure modes dominate:

- **Cost surprises.** A looping or retrying agent can burn through an
  expensive frontier model on work a small model would have done fine.
- **Brittle calls.** A single provider error kills the whole run.

budgetrouter addresses both with three small composable pieces: routing
rules that send each request to a suitable model, a fallback chain for when
a provider errors, and a budget gate that tracks cumulative estimated cost
and refuses calls past a cap. It drops into an existing app (LangGraph or
otherwise) as a library — it is not a framework and does not run your agent.

## Quickstart

No API keys needed: the quickstart runs against a built-in `FakeProvider`.

```python
from budgetrouter import BudgetGate, FakeProvider, Route, RouteRule, Router

providers = {
    "fake": FakeProvider(
        {"fake-small": "4", "fake-large": "4 (with a full proof sketch)"},
    ),
}
router = Router(
    providers=providers,
    rules=[RouteRule(provider="fake", model="fake-large", tag="hard")],
    default=Route(provider="fake", model="fake-small"),
    fallbacks=[Route(provider="fake", model="fake-small")],
    budget=BudgetGate(budget_usd=1.00, per_request_cap_usd=0.05),
)

easy = router.complete([{"role": "user", "content": "What is 2 + 2?"}])
print(easy.text)  # -> "4", served by fake/fake-small

hard = router.complete(
    [{"role": "user", "content": "What is 2 + 2? Show your work."}], tag="hard"
)
print(hard.text)  # -> routed to fake-large by the "hard" tag rule
```

The same example lives in [`examples/quickstart.py`](examples/quickstart.py)
and runs as-is after `pip install -e .`.

## Concepts

- **Provider** — anything with a `name` and a
  `complete(messages, model) -> Completion` method (a `typing.Protocol`).
  `Completion` carries the text plus input/output token usage. Bundled
  adapters cover the OpenAI and Anthropic API shapes (see below);
  `FakeProvider` is the in-memory implementation for tests and demos, with
  canned responses, fixed token counts, and simulated failures.
- **Router** — holds providers, an ordered list of `RouteRule`s, a default
  route, and a fallback chain. Rules match on a caller-supplied tag (e.g.
  `"hard"`, `"summarize"`) and/or estimated input size, so cheap requests go
  to cheap models and hard or long ones go to stronger models. If the chosen
  provider raises `ProviderError`, the next fallback is tried.
- **BudgetGate** — tracks cumulative spend using an editable per-model price
  table (USD per 1K tokens). It checks an estimate before each call and
  records actual usage after. Exceeding the per-run budget — or the optional
  per-request cap — raises `BudgetExceeded`, which the router deliberately
  does *not* treat as a provider failure: being out of budget is terminal,
  not a reason to spend elsewhere.
- **Eval runner** — a tiny JSONL harness (`{"input", "target"}` per line)
  with exact-match and substring scoring so routing changes can be
  smoke-checked for quality next to their cost:

  ```console
  $ python -m budgetrouter.eval --demo
  [PASS] input='What is the capital of France?' target='Paris' output='Paris'
  [PASS] input='What is 2 + 2?' target='4' output='4'
  [PASS] input='What color is the sky on a clear day?' target='blue' output='The sky is blue.'
  3/3 passed (100%)
  ```

## Real providers

Two adapters are bundled. Both are implemented with the standard library
only (`urllib.request`) — no vendor SDKs — and both raise `ProviderError`
on HTTP errors or network failures, so the fallback chain works with real
providers exactly as it does with `FakeProvider`.

### OpenAI-compatible

`OpenAICompatibleProvider` talks to `POST {base_url}/chat/completions`, so
it works with OpenAI and with any compatible server (gateways, local model
servers). Keys come from the environment unless passed explicitly:

```console
$ export OPENAI_API_KEY=sk-...
```

```python
from budgetrouter import OpenAICompatibleProvider, Route, Router

providers = {
    "openai": OpenAICompatibleProvider(
        "https://api.openai.com/v1", model="gpt-4o-mini"
    ),
}
router = Router(providers=providers, default=Route("openai", "gpt-4o-mini"))
```

To read the key from a differently named variable, pass
`api_key_env="MY_KEY_VAR"`; to point at a compatible server, change
`base_url`. A `timeout` (seconds) can be set per provider.

### Anthropic

`AnthropicProvider` talks to `POST /v1/messages` (base URL defaults to
`https://api.anthropic.com`). System messages are split out of the message
list into Anthropic's top-level `system` field automatically, and text
content blocks plus `usage.input_tokens` / `usage.output_tokens` are parsed
into the usual `Completion`.

```console
$ export ANTHROPIC_API_KEY=sk-ant-...
```

```python
from budgetrouter import AnthropicProvider, Route, Router

providers = {
    "anthropic": AnthropicProvider(model="claude-sonnet-4-5"),
}
router = Router(
    providers=providers, default=Route("anthropic", "claude-sonnet-4-5")
)
```

Mixing providers in one router — e.g. OpenAI as the default route and
Anthropic as the fallback — is the normal case; the router and budget gate
do not care which API shape served a call. Before relying on the budget
gate with real models, check the price table in
`budgetrouter.DEFAULT_PRICES`: the shipped vendor entries are illustrative
defaults, so verify them against the vendor's current pricing page.

### Verifying against a real API

Both adapters are covered by integration tests against a local stub
server that speaks the real API shapes (see `tests/test_adapters.py`).
To verify against a real endpoint, use
[`examples/live_smoke_test.py`](examples/live_smoke_test.py): point it
at any OpenAI-compatible endpoint with a real key (it defaults to
GitHub Models' free tier) and it exercises the adapter plus a
router + budget round-trip. The Anthropic adapter has *not* yet been
verified against Anthropic's live API (there is no free tier for the
native API); it follows the documented request/response format and the
stub tests.

## Async

Agent code is async by default, so budgetrouter ships a full async API:
`AsyncRouter`, `AsyncFakeProvider`, and async twins of both real
adapters (`AsyncOpenAICompatibleProvider`, `AsyncAnthropicProvider`).
`AsyncRouter.complete` mirrors the sync router's semantics exactly —
same routing rules, same fallback chain, same budget behavior
(`BudgetExceeded` still propagates immediately and is never treated as
a provider failure). One deliberate addition: when a budget gate is
attached, each attempt's check → call → record cycle runs under a lock,
so concurrent calls can never race past the cap and recorded spend is a
hard ceiling. Without a budget, calls run fully concurrently.

```python
import asyncio

from budgetrouter import (
    AsyncFakeProvider,
    AsyncRouter,
    BudgetGate,
    Route,
    RouteRule,
)

async def main() -> None:
    providers = {
        "fake": AsyncFakeProvider(
            {"fake-small": "4", "fake-large": "4 (with a full proof sketch)"},
        ),
    }
    router = AsyncRouter(
        providers=providers,
        rules=[RouteRule(provider="fake", model="fake-large", tag="hard")],
        default=Route(provider="fake", model="fake-small"),
        fallbacks=[Route(provider="fake", model="fake-small")],
        budget=BudgetGate(budget_usd=1.00, per_request_cap_usd=0.05),
    )
    easy = await router.complete([{"role": "user", "content": "What is 2 + 2?"}])
    hard = await router.complete(
        [{"role": "user", "content": "What is 2 + 2? Show your work."}],
        tag="hard",
    )
    print(easy.text, "|", hard.text)

asyncio.run(main())
```

Two honesty notes. First, the async adapters are async at the event-loop
level, not at the socket level: the package has zero dependencies, so
there is no async HTTP client — the blocking `urllib` call runs via
`asyncio.to_thread`. Concurrent calls genuinely overlap and the loop
never blocks, at the cost of one worker thread per in-flight request.
Second, the sync `Router` also has an `acomplete` method with the same
semantics, which runs a provider's native `acomplete` when it defines
one and bridges a sync-only provider through a worker thread — existing
providers keep working in async apps unchanged.

[`examples/langgraph_integration.py`](examples/langgraph_integration.py)
puts `AsyncRouter` inside a LangGraph `StateGraph` — a classify node on
the cheap model, an answer node on the strong one, one budget gate per
run — and runs key-free against `AsyncFakeProvider` (langgraph itself is
an example-only dependency; budgetrouter does not depend on it).
[`examples/langgraph_example.py`](examples/langgraph_example.py) shows
the same idea using `Router.acomplete` in a single node.

## Demo: an agent workload

[`examples/agent_workload_demo.py`](examples/agent_workload_demo.py) runs
the full stack — real adapters, routing, fallback, budget gate — against
local stub servers, so it works offline with no API keys. It simulates ten
mixed tasks tagged `cheap`/`strong`; partway through, the strong provider
starts returning HTTP 500 and the fallback visibly takes over, while spend
climbs toward the budget cap:

```console
$ python examples/agent_workload_demo.py
...
--- summary ---
tasks completed : 10 (2 served by fallback)
total spend     : $0.015234 of $0.017 cap (90% used, $0.001766 left)
per-model breakdown:
  gpt-4o       4 calls, 3000 tokens, $0.015000
  gpt-4o-mini  6 calls, 840 tokens, $0.000234
```

## How this differs from LiteLLM / hosted gateways

- **In-process library, not a proxy or a mega-SDK.** Hosted gateways
  (Portkey, LiteLLM Proxy, OpenRouter-style services) put a network hop
  and a running service between your code and the model. LiteLLM the SDK
  is in-process, but its center of gravity is breadth: 100+ provider
  adapters behind one interface. budgetrouter is a few hundred lines you
  import; the routing, fallback, and budget logic runs in your process,
  under your tests, with no extra service to deploy or keep up.
- **Budget enforcement is a first-class API, not a dashboard setting.**
  The gate is consulted before every call and updated after it, in code:
  per-run and per-request caps raise `BudgetExceeded` in the call path,
  so an agent loop stops instead of overspending. In gateway products,
  budgets typically live in admin configuration and alerting, outside
  the program's control flow.
- **Two API shapes, deliberately.** The bundled adapters cover the
  OpenAI chat-completions shape (which many providers and local servers
  speak) and Anthropic's messages shape. Anything else can implement
  the small `Provider` / `AsyncProvider` protocol directly — the router
  does not care how a provider talks.
- **Zero dependencies.** Standard library only, including the async
  adapters (see the Async section for exactly what that means).
- **What it does not do:** it is not an agent framework, not a hosted
  control plane (no dashboards, key management, or observability
  pipeline), and not a billing system. If you need those, use a gateway
  product — possibly with budgetrouter inside the agent process as well.

## Features

- Ordered routing rules by tag and estimated input size, with a default route
- Fallback chain across providers/models on provider errors
- Budget gate with per-run and per-request caps and an editable price table
- Bundled OpenAI-compatible and Anthropic adapters (stdlib only, env-var keys)
- Full async API (`AsyncRouter`, `AsyncFakeProvider`, async adapters) with
  a lock-guarded budget gate, so concurrent calls cannot overshoot the cap
- Provider protocol + `FakeProvider` for tests, demos, and CI
- Minimal eval runner (`exact` / `includes` scoring, JSONL datasets)
- Zero runtime dependencies (standard library only), typed, Python ≥ 3.10

## Non-goals

- **Not an agent framework.** No agents, tools, memory, planning, or
  orchestration — budgetrouter only decides *which model serves a call* and
  *whether the budget allows it*. Use it inside LangGraph, a hand-rolled
  loop, or a plain script.
- No provider SDK dependencies. The bundled adapters speak plain HTTPS via
  the standard library; provider-specific extras (streaming, tool use,
  batch APIs) are out of scope, and anything unusual can still implement
  the `Provider` protocol directly.
- No precise tokenization. Input size and pre-call costs are estimates
  (chars / 4); accounting uses the provider's reported usage.
- Not a billing system. The budget gate is a safety stop, not invoicing.

## Roadmap

- Streaming completions
- Precise tokenizers for pre-call estimates (per-provider, opt-in)
- Cost reporting helpers (per-tag / per-model spend breakdowns)
- Weighted and latency-aware routing strategies

## Development

```console
$ python -m venv .venv && source .venv/bin/activate
$ pip install -e ".[dev]"
$ pytest
```

## License

MIT — see [LICENSE](LICENSE).
