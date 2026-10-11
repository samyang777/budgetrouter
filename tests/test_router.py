import pytest

from budgetrouter import (
    BudgetExceeded,
    BudgetGate,
    FakeProvider,
    ProviderError,
    Route,
    RouteRule,
    Router,
)

MESSAGES = [{"role": "user", "content": "hello"}]


def make_router(**overrides):
    providers = {
        "fake": FakeProvider({"fake-small": "small answer", "fake-large": "large answer"}),
    }
    kwargs = dict(
        providers=providers,
        rules=[RouteRule(provider="fake", model="fake-large", tag="hard")],
        default=Route(provider="fake", model="fake-small"),
    )
    kwargs.update(overrides)
    return Router(**kwargs)


def test_default_route_used_without_tag():
    out = make_router().complete(MESSAGES)
    assert out.text == "small answer"
    assert out.model == "fake-small"


def test_tag_rule_selects_large_model():
    out = make_router().complete(MESSAGES, tag="hard")
    assert out.text == "large answer"
    assert out.model == "fake-large"


def test_size_rule_selects_by_estimated_input():
    router = make_router(
        rules=[RouteRule(provider="fake", model="fake-large", min_input_tokens=100)],
    )
    small = router.complete(MESSAGES)
    big = router.complete([{"role": "user", "content": "x" * 800}])
    assert small.model == "fake-small"
    assert big.model == "fake-large"


def test_first_matching_rule_wins():
    router = make_router(
        rules=[
            RouteRule(provider="fake", model="fake-large", tag="hard"),
            RouteRule(provider="fake", model="fake-small", tag="hard"),
        ],
    )
    assert router.complete(MESSAGES, tag="hard").model == "fake-large"


def test_unknown_provider_in_route_rejected():
    with pytest.raises(ValueError, match="unknown providers"):
        Router(providers={}, default=Route(provider="nope", model="fake-small"))


def test_fallback_on_provider_failure():
    flaky = FakeProvider("primary answer", name="flaky", fail_times=1)
    backup = FakeProvider("backup answer", name="backup")
    router = Router(
        providers={"flaky": flaky, "backup": backup},
        default=Route(provider="flaky", model="fake-small"),
        fallbacks=[Route(provider="backup", model="fake-small")],
    )
    out = router.complete(MESSAGES)
    assert out.text == "backup answer"
    assert out.provider == "backup"
    assert len(flaky.calls) == 1
    assert len(backup.calls) == 1


def test_all_routes_failing_raises_provider_error():
    router = Router(
        providers={
            "a": FakeProvider(name="a", fail_times=5),
            "b": FakeProvider(name="b", fail_times=5),
        },
        default=Route(provider="a", model="fake-small"),
        fallbacks=[Route(provider="b", model="fake-small")],
    )
    with pytest.raises(ProviderError, match="all routes failed"):
        router.complete(MESSAGES)


def test_budget_exceeded_is_not_treated_as_provider_failure():
    # The gate blocks the call outright; fallback must not silently spend elsewhere.
    backup = FakeProvider("backup answer", name="backup")
    router = Router(
        providers={"fake": FakeProvider("x"), "backup": backup},
        default=Route(provider="fake", model="fake-large"),
        fallbacks=[Route(provider="backup", model="fake-small")],
        budget=BudgetGate(budget_usd=0.0001),
    )
    with pytest.raises(BudgetExceeded):
        router.complete(MESSAGES)
    assert backup.calls == []
