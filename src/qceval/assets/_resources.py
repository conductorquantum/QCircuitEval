"""Centralized importlib.resources access for packaged qceval assets.

Runtime loaders should resolve packaged JSON/JSONL through these helpers so
package-name strings and path layouts stay in one place.
"""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable

_KNOWN_SUITES = frozenset({"core", "qec"})


def asset_root() -> Traversable:
    """Return the packaged ``qceval.assets`` root.

    Returns:
        Traversable for the installed assets package.
    """

    return resources.files("qceval.assets")


def asset_path(*parts: str) -> Traversable:
    """Return a Traversable under ``qceval.assets``.

    Args:
        *parts: Path segments relative to the assets package root.

    Returns:
        Traversable for the selected asset path.
    """

    return asset_root().joinpath(*parts)


def read_text(*parts: str) -> str:
    """Read a UTF-8 text asset relative to ``qceval.assets``.

    Args:
        *parts: Path segments under the assets package.

    Returns:
        Decoded asset text.

    Raises:
        FileNotFoundError: If the asset path is missing from the package.
    """

    return asset_path(*parts).read_text(encoding="utf-8")


def read_bytes(*parts: str) -> bytes:
    """Read a binary asset relative to ``qceval.assets``.

    Args:
        *parts: Path segments under the assets package.

    Returns:
        Raw asset bytes.

    Raises:
        FileNotFoundError: If the asset path is missing from the package.
    """

    return asset_path(*parts).read_bytes()


def task_resource(suite: str, framework: str) -> Traversable:
    """Return the JSONL task asset for one suite/framework pair.

    Args:
        suite: Benchmark suite name (``core`` or ``qec``).
        framework: Framework asset stem (for example ``qiskit``).

    Returns:
        Traversable pointing at ``{suite}/{framework}.jsonl``.

    Raises:
        ValueError: If ``suite`` is not a packaged suite name.
    """

    _require_suite(suite)
    return asset_path(suite, f"{framework}.jsonl")


def contract_resource(suite: str) -> Traversable:
    """Return the JSONL contract registry for one suite.

    Args:
        suite: Benchmark suite name (``core`` or ``qec``).

    Returns:
        Traversable pointing at ``contracts/{suite}.jsonl``.

    Raises:
        ValueError: If ``suite`` is not a packaged suite name.
    """

    _require_suite(suite)
    return asset_path("contracts", f"{suite}.jsonl")


def target_resource(suite: str, name: str) -> Traversable:
    """Return one file under a suite's packaged target directory.

    Args:
        suite: Benchmark suite name (``core`` or ``qec``).
        name: Filename within ``targets/{suite}/``.

    Returns:
        Traversable pointing at ``targets/{suite}/{name}``.

    Raises:
        ValueError: If ``suite`` is not a packaged suite name.
    """

    _require_suite(suite)
    return asset_path("targets", suite, name)


def _require_suite(suite: str) -> None:
    if suite not in _KNOWN_SUITES:
        raise ValueError(f"unknown suite: {suite}")
