from scripts.run_openrouter_canaries import _sentinel_matches


def test_canary_requires_exact_sentinel_return_value() -> None:
    raw = {"choices": [{"message": {"content": "ROUTE_CANARY_123456"}}]}
    assert _sentinel_matches(raw, 123456) is True
    assert _sentinel_matches(raw, 654321) is False


def test_canary_rejects_missing_or_malformed_sentinel_code() -> None:
    assert _sentinel_matches(None, 123456) is False
    assert _sentinel_matches({}, 123456) is False
    assert _sentinel_matches({"choices": [{"message": {"content": None}}]}, 123456) is False
