"""Read and write QCircuitEval run output files.

JSON output stores a complete run payload.  JSONL output stores one result per
line plus a final summary line, which supports streaming, resume, and partial
inspection while long provider runs are still active.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from qceval.models import BenchmarkRecord
from qceval.serialization import to_jsonable

OutputFormat = Literal["auto", "json", "jsonl"]


def write_output(path: Path, payload: dict[str, Any], output_format: OutputFormat = "auto") -> None:
    """Write run payload as JSON or JSONL.

    Args:
        path: Destination file path.  Parent directories are created.
        payload: Complete run payload returned by
            :class:`qceval.core.runner.BenchmarkRunner`.
        output_format: Explicit format or ``"auto"`` to infer from suffix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = infer_format(path, output_format)
    if resolved == "jsonl":
        _write_jsonl(path, payload)
        return
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def infer_format(path: Path, output_format: OutputFormat) -> Literal["json", "jsonl"]:
    """Resolve output format from explicit option and path.

    Args:
        path: Output path used when ``output_format`` is ``"auto"``.
        output_format: Requested output format.

    Returns:
        ``"jsonl"`` for ``.jsonl`` paths under auto mode, otherwise ``"json"``.
    """
    if output_format != "auto":
        return output_format
    return "jsonl" if path.suffix == ".jsonl" else "json"


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    lines = []
    for record in payload["results"]:
        lines.append(json.dumps({"kind": "result", **to_jsonable(record)}, sort_keys=True))
    summary = _summary_payload(payload)
    lines.append(json.dumps({"kind": "summary", **to_jsonable(summary)}, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class JsonlRunWriter:
    """Append run records and summaries to a JSONL file.

    ``JsonlRunWriter`` uses a persistent file handle to avoid per-record
    open/close overhead.  It flushes and fsyncs every ``sync_interval``
    records so interrupted runs lose at most that many in-flight results.

    Args:
        path: Destination JSONL path.
        truncate: Whether to clear existing content during construction.
        sync_interval: Number of records between flush+fsync calls.  The
            default of ``1`` preserves per-record durability.  Higher values
            reduce I/O overhead for large runs at the cost of losing up to
            ``sync_interval - 1`` records on crash.

    Attributes:
        path: Destination JSONL path.
    """

    def __init__(self, path: Path, *, truncate: bool = True, sync_interval: int = 1) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if truncate else "a"
        self._handle = self.path.open(mode, encoding="utf-8")
        self._sync_interval = max(1, sync_interval)
        self._since_sync = 0

    def append(self, record: BenchmarkRecord) -> None:
        """Append one benchmark record.

        Args:
            record: Completed benchmark record.
        """
        self._append_line({"kind": "result", **record.to_dict()})

    def append_dict(self, record: dict[str, Any]) -> None:
        """Append one pre-serialized benchmark record.

        Args:
            record: Dictionary payload for a completed result.
        """
        self._append_line({"kind": "result", **to_jsonable(record)})

    def finalize(self, payload: dict[str, Any]) -> None:
        """Append final summary line for a run payload.

        Args:
            payload: Complete run payload containing top-level metadata and
                ``summary`` fields.
        """
        self._sync()
        summary = _summary_payload(payload)
        self._handle.write(json.dumps(to_jsonable(summary | {"kind": "summary"}), sort_keys=True) + "\n")
        self._sync()

    def close(self) -> None:
        """Flush, sync, and close the underlying file handle."""
        if not self._handle.closed:
            self._sync()
            self._handle.close()

    def __enter__(self) -> JsonlRunWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _append_line(self, payload: dict[str, Any]) -> None:
        self._handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")
        self._since_sync += 1
        if self._since_sync >= self._sync_interval:
            self._sync()

    def _sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._since_sync = 0


def read_completed(path: Path) -> dict[tuple[str, str, str, int, int], dict[str, Any]]:
    """Read completed result records from a JSON or JSONL run file.

    JSON envelopes contribute every object in ``results``. JSONL files ignore
    malformed lines and summary lines. The returned mapping is keyed by
    ``(suite, framework, task_id, sample_index, attempt_index)``.

    Args:
        path: JSON envelope or JSONL file produced by :class:`JsonlRunWriter`
            or :func:`write_output`.

    Returns:
        Mapping from ``(suite, framework, task_id, sample_index, attempt_index)``
        to serialized record payload.
    """
    completed: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    if not path.exists():
        return completed
    text = path.read_text(encoding="utf-8")
    envelope = _run_envelope(text)
    if envelope is not None:
        for payload in envelope["results"]:
            if isinstance(payload, dict):
                _store_completed_record(completed, payload, require_result_kind=False)
        return completed
    for line in text.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            _store_completed_record(completed, payload, require_result_kind=True)
    return completed


def _run_envelope(text: str) -> dict[str, Any] | None:
    if not text.lstrip().startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload
    return None


def _store_completed_record(
    completed: dict[tuple[str, str, str, int, int], dict[str, Any]],
    payload: dict[str, Any],
    *,
    require_result_kind: bool,
) -> None:
    kind = payload.get("kind")
    if require_result_kind:
        if kind != "result":
            return
    elif kind not in (None, "result"):
        return
    framework = payload.get("framework")
    task_id = payload.get("task_id")
    suite = payload.get("suite", "core")
    try:
        sample_index = _int_field(payload.get("sample_index", 0), "sample_index")
        attempt_index = _int_field(payload.get("attempt_index", 0), "attempt_index")
    except ValueError:
        return
    if isinstance(framework, str) and isinstance(task_id, str):
        completed[(str(suite), framework, task_id, sample_index, attempt_index)] = {
            key: value for key, value in payload.items() if key != "kind"
        }


def read_run_identity(path: Path) -> tuple[dict[str, Any] | None, set[str]]:
    """Read the full summary identity and per-record identity digests.

    Args:
        path: JSONL run artifact to inspect.

    Returns:
        Summary identity, when present, and all record-level identity digests.
    """
    identity: dict[str, Any] | None = None
    digests: set[str] = set()
    if not path.exists():
        return identity, digests
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("kind") == "summary" and isinstance(payload.get("run_identity"), dict):
            candidate = payload["run_identity"]
            if identity is not None and candidate != identity:
                raise ValueError("resume data contains conflicting run identities")
            identity = candidate
        lineage = payload.get("lineage")
        if isinstance(lineage, dict) and isinstance(lineage.get("run_identity_sha256"), str):
            digests.add(lineage["run_identity_sha256"])
    return identity, digests


def _int_field(value: Any, name: str) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None


def _summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {key: payload[key] for key in ("schema_version", "provider", "model", "qceval", "summary")}
    if "configuration_id" in payload:
        summary["configuration_id"] = payload["configuration_id"]
    if "run_id" in payload:
        summary["run_id"] = payload["run_id"]
    if "run_identity" in payload:
        summary["run_identity"] = payload["run_identity"]
    summary["suites"] = payload.get("suites", ["core"])
    return summary
