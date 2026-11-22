import pytest

from budgetrouter import (
    BudgetExceeded,
    BudgetGate,
    FakeProvider,
    FallbackPolicy,
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
    with pytest.raises(BudgetExceeded) as exc_info:
        router.complete(MESSAGES)
    assert backup.calls == []
    assert len(exc_info.value.attempts) == 1
    assert exc_info.value.attempts[0].outcome == "budget_exceeded"


def test_timeout_falls_back_and_records_attempt_log():
    slow = FakeProvider("primary", name="primary", fail_times=1, fail_kind="timeout")
    backup = FakeProvider("backup answer", name="backup")
    router = Router(
        providers={"primary": slow, "backup": backup},
        default=Route(provider="primary", model="fake-large"),
        fallbacks=[Route(provider="backup", model="fake-small")],
    )
    out = router.complete(MESSAGES)
    assert out.text == "backup answer"
    assert len(out.attempts) == 2
    assert out.attempts[0].outcome == "timeout"
    assert out.attempts[0].is_primary is True
    assert out.attempts[0].latency_ms >= 0.0
    assert out.attempts[1].outcome == "success"
    assert out.attempts[1].is_primary is False
    assert out.attempts[1].downgraded is True


def test_rate_limit_does_not_retry_primary_by_default():
    limited = FakeProvider(
        "primary", name="primary", fail_times=1, fail_kind="rate_limit", fail_status_code=429
    )
    backup = FakeProvider("backup answer", name="backup")
    router = Router(
        providers={"primary": limited, "backup": backup},
        default=Route(provider="primary", model="fake-large"),
        fallbacks=[Route(provider="backup", model="fake-small")],
    )
    out = router.complete(MESSAGES)
    assert out.provider == "backup"
    assert len(limited.calls) == 1  # not retried before fallback
    assert out.attempts[0].outcome == "rate_limit"
    assert out.attempts[0].status_code == 429
    assert out.attempts[1].outcome == "success"


def test_server_error_falls_back_to_smaller_model():
    broken = FakeProvider(
        "primary", name="primary", fail_times=1, fail_kind="server_error", fail_status_code=503
    )
    small = FakeProvider("small answer", name="small")
    router = Router(
        providers={"primary": broken, "small": small},
        default=Route(provider="primary", model="fake-large"),
        fallbacks=[Route(provider="small", model="fake-small")],
    )
    out = router.complete(MESSAGES)
    assert out.model == "fake-small"
    assert out.attempts[0].outcome == "server_error"
    assert out.attempts[1].outcome == "success"


def test_auth_failure_aborts_immediately_without_fallback():
    bad_key = FakeProvider(
        "primary", name="primary", fail_times=1, fail_kind="auth", fail_status_code=401
    )
    backup = FakeProvider("backup answer", name="backup")
    router = Router(
        providers={"primary": bad_key, "backup": backup},
        default=Route(provider="primary", model="fake-large"),
        fallbacks=[Route(provider="backup", model="fake-small")],
    )
    with pytest.raises(ProviderError) as exc_info:
        router.complete(MESSAGES)
    assert exc_info.value.kind == "auth"
    assert exc_info.value.status_code == 401
    assert backup.calls == []
    assert len(exc_info.value.attempts) == 1
    assert exc_info.value.attempts[0].outcome == "auth"


def test_auth_policy_can_skip_same_provider_and_use_other_provider():
    bad_openai = FakeProvider(
        "x", name="openai", fail_times=5, fail_kind="auth", fail_status_code=401
    )
    anthropic = FakeProvider("anthropic answer", name="anthropic")
    policy = FallbackPolicy(
        actions={"auth": "fallback"}, skip_same_provider_on_auth=True
    )
    router = Router(
        providers={"openai": bad_openai, "anthropic": anthropic},
        default=Route(provider="openai", model="gpt-4o"),
        fallbacks=[
            Route(provider="openai", model="gpt-4o-mini"),
            Route(provider="anthropic", model="claude-haiku-4-5"),
        ],
        fallback_policy=policy,
    )
    out = router.complete(MESSAGES)
    assert out.provider == "anthropic"
    assert len(bad_openai.calls) == 1  # skipped gpt-4o-mini on same provider
    assert len(out.attempts) == 2


def test_retry_then_fallback_policy_retries_primary_once():
    flaky = FakeProvider("recovered", name="primary", fail_times=1, fail_kind="timeout")
    backup = FakeProvider("backup", name="backup")
    router = Router(
        providers={"primary": flaky, "backup": backup},
        default=Route(provider="primary", model="fake-large"),
        fallbacks=[Route(provider="backup", model="fake-small")],
        fallback_policy=FallbackPolicy(actions={"timeout": "retry_then_fallback"}, max_retries=1),
    )
    out = router.complete(MESSAGES)
    assert out.text == "recovered"
    assert out.provider == "primary"
    assert len(flaky.calls) == 2
    assert backup.calls == []
    assert [a.outcome for a in out.attempts] == ["timeout", "success"]
    assert out.attempts[1].downgraded is False


def test_all_routes_failing_reports_all_failure_kinds_and_attempts():
    router = Router(
        providers={
            "a": FakeProvider(name="a", fail_times=5, fail_kind="timeout"),
            "b": FakeProvider(name="b", fail_times=5, fail_kind="server_error", fail_status_code=500),
        },
        default=Route(provider="a", model="fake-large"),
        fallbacks=[Route(provider="b", model="fake-small")],
    )
    with pytest.raises(ProviderError) as exc_info:
        router.complete(MESSAGES)
    msg = str(exc_info.value)
    assert "timeout" in msg
    assert "server_error" in msg
    assert [a.outcome for a in exc_info.value.attempts] == ["timeout", "server_error"]


def test_unclassified_provider_error_still_falls_back():
    class LegacyProvider:
        name = "legacy"

        def complete(self, messages, model):
            raise ProviderError("plain boom")

    backup = FakeProvider("backup ok", name="backup")
    router = Router(
        providers={"legacy": LegacyProvider(), "backup": backup},
        default=Route(provider="legacy", model="fake-large"),
        fallbacks=[Route(provider="backup", model="fake-small")],
    )
    out = router.complete(MESSAGES)
    assert out.text == "backup ok"
    assert out.attempts[0].outcome == "unknown"


def test_fallback_policy_rejects_unknown_action():
    with pytest.raises(ValueError, match="unknown fallback action"):
        FallbackPolicy(actions={"timeout": "magic"})
