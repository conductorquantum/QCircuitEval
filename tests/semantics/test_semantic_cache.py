"""Unit tests for the content-addressed semantic cache."""

from __future__ import annotations

from pathlib import Path

from qceval.semantics.cache import ContentAddressedCache, SemanticCacheKey


def _key(suffix: str = "a") -> SemanticCacheKey:
    return SemanticCacheKey(
        candidate_hash=f"cand-{suffix}",
        contract_hash=f"contract-{suffix}",
        target_hash=f"target-{suffix}",
        ir_version="1",
        verifier_version="1",
        framework="qiskit",
        framework_version="2",
        backend=None,
        precision="float64",
        parameters_hash="p",
        observation_hash="o",
        limits_hash="l",
    )


def test_content_addressed_cache_round_trip(tmp_path: Path) -> None:
    cache = ContentAddressedCache(tmp_path)
    key = _key()
    assert cache.get(key).hit is False
    path = cache.put(key, {"status": "verified_pass", "metric": 0.0})
    assert path.is_file()
    lookup = cache.get(key)
    assert lookup.hit is True
    assert lookup.payload is not None
    assert lookup.payload["status"] == "verified_pass"


def test_content_addressed_cache_isolates_keys(tmp_path: Path) -> None:
    cache = ContentAddressedCache(tmp_path)
    cache.put(_key("a"), {"status": "a"})
    assert cache.get(_key("b")).hit is False
