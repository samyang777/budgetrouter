"""budgetrouter inside a LangGraph agent: two nodes, two models, one budget.

A small StateGraph with a classify step and an answer step. The classify
node's model call is routed to a cheap model, the answer node's to a
strong one — both go through a single ``AsyncRouter``, and a
``BudgetGate`` created fresh for each run caps what the whole run may
spend. Each node records which route served it, and the run ends by
printing the route log and the total spend.

Everything runs offline against ``AsyncFakeProvider``: no API keys, no
network. Swap the fake for ``AsyncOpenAICompatibleProvider`` /
``AsyncAnthropicProvider`` and the graph is unchanged.

Requirements (example-only — budgetrouter itself has no dependencies):

    pip install -e .          # budgetrouter
    pip install langgraph     # only needed to run THIS example
    python examples/langgraph_integration.py
"""

import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from budgetrouter import (
    AsyncFakeProvider,
    AsyncRouter,
    BudgetGate,
    Route,
    RouteRule,
)

RUN_BUDGET_USD = 0.01


class AgentState(TypedDict, total=False):
    question: str
    category: str
    answer: str
    route_log: list[str]


def build_router() -> AsyncRouter:
    """One router for the whole graph; a fresh BudgetGate per run."""
    providers = {
        "fake": AsyncFakeProvider(
            {
                "fake-small": "math",
                "fake-large": "4 — two plus two equals four.",
            }
        ),
    }
    return AsyncRouter(
        providers=providers,
        rules=[
            RouteRule(provider="fake", model="fake-small", tag="classify"),
            RouteRule(provider="fake", model="fake-large", tag="answer"),
        ],
        default=Route(provider="fake", model="fake-small"),
        budget=BudgetGate(budget_usd=RUN_BUDGET_USD, per_request_cap_usd=0.005),
    )


def make_nodes(router: AsyncRouter):
    """Build the graph's nodes around one router instance."""

    async def classify(state: AgentState) -> dict:
        completion = await router.complete(
            [
                {
                    "role": "user",
                    "content": f"Classify this question: {state['question']}",
                }
            ],
            tag="classify",
        )
        entry = (
            f"classify -> {completion.provider}/{completion.model}"
        )
        return {
            "category": completion.text,
            "route_log": state.get("route_log", []) + [entry],
        }

    async def answer(state: AgentState) -> dict:
        completion = await router.complete(
            [
                {
                    "role": "user",
                    "content": (
                        f"Category: {state['category']}. "
                        f"Answer this question: {state['question']}"
                    ),
                }
            ],
            tag="answer",
        )
        entry = f"answer -> {completion.provider}/{completion.model}"
        return {
            "answer": completion.text,
            "route_log": state.get("route_log", []) + [entry],
        }

    return classify, answer


def build_graph(router: AsyncRouter):
    classify, answer = make_nodes(router)
    return (
        StateGraph(AgentState)
        .add_node("classify", classify)
        .add_node("answer", answer)
        .add_edge(START, "classify")
        .add_edge("classify", "answer")
        .add_edge("answer", END)
        .compile()
    )


async def main() -> None:
    questions = [
        "What is 2 + 2?",
        "What is 17 + 25? Show the steps.",
    ]
    for question in questions:
        router = build_router()  # fresh router + fresh budget gate per run
        graph = build_graph(router)
        out = await graph.ainvoke({"question": question})
        print(f"question : {question}")
        print(f"category : {out['category']}")
        print(f"answer   : {out['answer']}")
        for entry in out["route_log"]:
            print(f"  route    {entry}")
        gate = router.budget
        print(f"spend    : ${gate.spent:.6f} of ${gate.budget_usd:.2f} budget\n")


if __name__ == "__main__":
    asyncio.run(main())
