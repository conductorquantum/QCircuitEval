"""Contract registry loading, lookup, hashing, and diffing."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qceval.assets._resources import contract_resource
from qceval.semantics.contracts.kinds import Contract, ContractValidationError
from qceval.semantics.contracts.serialization import contract_hash, parse_contract_json

ChangeKind = Literal["added", "removed", "modified"]


@dataclass(frozen=True)
class ContractChange:
    """One stable-key difference between two registries."""

    kind: ChangeKind
    suite: str
    task_id: str
    old_version: str | None
    new_version: str | None
    old_hash: str | None
    new_hash: str | None


class ContractRegistry:
    """Immutable registry of unique validated task contracts."""

    def __init__(self, contracts: Iterable[Contract]) -> None:
        """Initialize a registry and reject duplicate task keys.

        Args:
            contracts: Validated contracts in any order.

        Raises:
            ContractValidationError: If two contracts share a registry key.
        """

        by_key: dict[tuple[str, str], Contract] = {}
        for contract in contracts:
            if contract.key in by_key:
                raise ContractValidationError("$", f"duplicate contract key {contract.key!r}")
            by_key[contract.key] = contract
        self._by_key = dict(sorted(by_key.items()))

    @classmethod
    def from_jsonl(cls, payload: str | bytes) -> ContractRegistry:
        """Parse a strict JSONL contract registry.

        Args:
            payload: UTF-8 JSONL text or bytes.

        Returns:
            Parsed immutable registry.
        """

        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        contracts = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                contracts.append(parse_contract_json(line))
            except ContractValidationError as exc:
                raise ContractValidationError(f"line[{line_number}]{exc.path}", exc.reason) from exc
        return cls(contracts)

    @classmethod
    def from_path(cls, path: Path) -> ContractRegistry:
        """Load a registry from a UTF-8 JSONL path.

        Args:
            path: Contract registry path.

        Returns:
            Parsed immutable registry.
        """

        return cls.from_jsonl(path.read_text(encoding="utf-8"))

    @classmethod
    def from_package(cls, suite: str = "core") -> ContractRegistry:
        """Load a packaged registry for ``suite``.

        Args:
            suite: Packaged contract registry name (``core`` or ``qec``).

        Returns:
            Loaded registry.

        Raises:
            ValueError: If ``suite`` is not a packaged suite name.
            FileNotFoundError: If the packaged registry asset is missing.
            ContractValidationError: If any registry line fails validation.
        """

        return cls.from_jsonl(contract_resource(suite).read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self) -> Iterator[Contract]:
        return iter(self._by_key.values())

    def get(self, suite: str, task_id: str) -> Contract:
        """Return one contract by normalized stable key.

        Args:
            suite: Benchmark suite.
            task_id: Task identifier, normalized for the core suite.

        Returns:
            Matching contract.
        """

        key = (suite, str(task_id).zfill(2) if suite == "core" else str(task_id))
        return self._by_key[key]

    def hashes(self) -> Mapping[tuple[str, str], str]:
        """Return content hashes keyed by ``(suite, task_id)``."""

        return {key: contract_hash(contract) for key, contract in self._by_key.items()}

    def diff(self, other: ContractRegistry) -> tuple[ContractChange, ...]:
        """Return deterministic key/hash differences from this registry.

        Args:
            other: Registry representing the new state.

        Returns:
            Ordered added, removed, and modified contract changes.
        """

        changes = []
        keys = sorted(set(self._by_key) | set(other._by_key))
        for key in keys:
            old = self._by_key.get(key)
            new = other._by_key.get(key)
            change = _contract_change(key, old, new)
            if change is not None:
                changes.append(change)
        return tuple(changes)


def _contract_change(key: tuple[str, str], old: Contract | None, new: Contract | None) -> ContractChange | None:
    old_hash = None if old is None else contract_hash(old)
    new_hash = None if new is None else contract_hash(new)
    if old_hash == new_hash:
        return None
    if old is None:
        kind: ChangeKind = "added"
    elif new is None:
        kind = "removed"
    else:
        kind = "modified"
    return ContractChange(
        kind=kind,
        suite=key[0],
        task_id=key[1],
        old_version=None if old is None else old.contract_version,
        new_version=None if new is None else new.contract_version,
        old_hash=old_hash,
        new_hash=new_hash,
    )
