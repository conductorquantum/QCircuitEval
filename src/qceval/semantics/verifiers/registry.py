"""Immutable semantic verifier engine registry."""

from __future__ import annotations

from collections.abc import Iterable

from qceval.semantics.verifiers.base import VerifierEngine


class VerifierRegistry:
    """Unique engine registry keyed by descriptor name."""

    def __init__(self, engines: Iterable[VerifierEngine]) -> None:
        """Initialize the registry and reject duplicate names.

        Args:
            engines: Semantic verifier engines.
        """
        values: dict[str, VerifierEngine] = {}
        for engine in engines:
            name = engine.descriptor().name
            if not name or name in values:
                raise ValueError(f"duplicate or empty verifier engine name {name!r}")
            values[name] = engine
        self._engines = dict(sorted(values.items()))

    def get(self, name: str) -> VerifierEngine | None:
        """Return an engine or ``None`` when the capability is absent.

        Args:
            name: Contract route engine name.

        Returns:
            Registered engine or ``None``.
        """
        return self._engines.get(name)
