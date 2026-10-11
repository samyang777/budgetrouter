"""budgetrouter inside a LangGraph node — routing + budget, async-native.

LangGraph nodes are async in production agents, which is exactly what
``Router.acomplete`` is for: same routing rules, fallback chain, and
budget gate as the sync ``Router.complete``, without blocking the event
loop.

This example runs with no API keys (FakeProvider). If ``langgraph`` is
installed it builds a real one-node StateGraph; if not, it calls the
node function directly — the node is the part that matters, the graph
plumbing around it is interchangeable.

    pip install -e .            # budgetrouter itself needs nothing else
    pip install langgraph       # optional, only for the StateGraph part
    python examples/langgraph_example.py
"""

import asyncio

from budgetrouter import BudgetGate, FakeProvider, Route, RouteRule, Router

router = Router(
    providers={
        "fake": FakeProvider(
            {"fake-small": "4", "fake-large": "4 (with a full proof sketch)"},
        ),
    },
    rules=[RouteRule(provider="fake", model="fake-large", tag="hard")],
    default=Route(provider="fake", model="fake-small"),
    fallbacks=[Route(provider="fake", model="fake-small")],
    budget=BudgetGate(budget_usd=1.00, per_request_cap_usd=0.05),
)


async def answer_node(state):
    """A LangGraph node: route this step's model call through budgetrouter.

    State in, state out — the node reads ``question``/``tag`` and writes
    ``answer``. Which model served the call, and whether the budget would
    have allowed it at all, is budgetrouter's job, not the graph's.
    """
    completion = await router.acomplete(
        [{"role": "user", "content": state["question"]}],
        tag=state.get("tag"),
    )
    return {"answer": completion.text, "served_by": completion.model}


async def main() -> None:
    easy = {"question": "What is 2 + 2?"}
    hard = {"question": "What is 2 + 2? Show your work.", "tag": "hard"}

    try:
        from langgraph.graph import END, START, StateGraph
        from typing_extensions import TypedDict
    except ImportError:
        print("langgraph not installed — calling the node directly instead.\n")
        for state in (easy, hard):
            out = await answer_node(state)
            print(f"{state['question']!r} -> {out['answer']!r} (via {out['served_by']})")
    else:

        class State(TypedDict, total=False):
            question: str
            tag: str
            answer: str
            served_by: str

        graph = (
            StateGraph(State)
            .add_node("answer", answer_node)
            .add_edge(START, "answer")
            .add_edge("answer", END)
            .compile()
        )
        for state in (easy, hard):
            out = await graph.ainvoke(state)
            print(f"{state['question']!r} -> {out['answer']!r} (via {out['served_by']})")

    print(f"\nspent so far: ${router.budget.spent:.6f} of $1.00")


if __name__ == "__main__":
    asyncio.run(main())
