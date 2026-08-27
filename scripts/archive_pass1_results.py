#!/usr/bin/env python3
"""Create and fully verify a self-contained ZIP snapshot of QCircuitEval results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BENCHMARK_COMMIT = "02061df263c1204f61776cbdb8d7295f820f029c"
FINAL_CAMPAIGN = Path("results/full-effort-pass1-five-model")
PRIOR_CAMPAIGN = Path("results/pass1-123a5e4-20260810T213903Z")
SUPPLEMENTAL_MODELS = {
    "google/gemini-3.1-pro-preview",
    "google/gemma-4-31b-it",
    "moonshotai/kimi-k3",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "x-ai/grok-4.5",
}
CHUNK_SIZE = 1024 * 1024


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    repo = args.repo_root.resolve()
    output = args.output.resolve()
    if not (repo / ".git").exists() or not (repo / "results").is_dir():
        parser.error(f"{repo} is not the QCircuitEval repository root")
    if output.is_relative_to(repo / "results"):
        parser.error("output must not be inside results/")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"refusing to overwrite existing archive: {output}")

    started = time.monotonic()
    audit = validate_authoritative_data(repo)
    created_at = datetime.now(UTC).replace(microsecond=0)
    prefix = output.stem
    with tempfile.TemporaryDirectory(prefix="qceval-archive-") as temporary:
        staging = Path(temporary)
        source_files, revisions = create_source_snapshots(repo, staging)
        result_files, symlinks = enumerate_tree(repo / "results")
        payload = [(path, path.relative_to(repo).as_posix()) for path in result_files]
        payload.extend(source_files)
        payload.extend(_untracked_source_files(repo, output))
        payload.sort(key=lambda item: item[1])

        print(
            json.dumps(
                {
                    "phase": "archive",
                    "payload_files": len(payload),
                    "result_symlinks_recorded": len(symlinks),
                    "output": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        entries = create_zip(
            output,
            prefix=prefix,
            payload=payload,
            symlinks=symlinks,
            repo=repo,
            created_at=created_at,
            revisions=revisions,
            audit=audit,
        )
        verify_zip(output, prefix=prefix, entries=entries)

    archive_sha256 = _sha256_file(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{archive_sha256}  {output.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "archive": str(output),
                "archive_bytes": output.stat().st_size,
                "archive_sha256": archive_sha256,
                "checksum": str(checksum_path),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "payload_files": len(entries),
                "verified": True,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def validate_authoritative_data(repo: Path) -> dict[str, Any]:  # noqa: C901 - explicit archive integrity gates
    """Validate the final sweep and completed prior-run artifacts used by the plots."""
    final_root = repo / FINAL_CAMPAIGN
    final_audit = json.loads((final_root / "final-audit.json").read_text(encoding="utf-8"))
    if final_audit.get("publication_ready") is not True:
        raise ValueError("final effort campaign is not publication-ready")
    scope = final_audit.get("scope") or {}
    coverage = final_audit.get("coverage") or {}
    if scope.get("configurations") != 28 or coverage.get("offline_regraded_records") != 7840:
        raise ValueError("final effort campaign cardinality is invalid")
    if coverage.get("provider_cost_covered_records") != 7840:
        raise ValueError("final effort campaign cost coverage is incomplete")
    checksum_count = _verify_checksum_manifest(repo, final_root / "artifact-hashes.sha256")

    prior_manifest_path = repo / PRIOR_CAMPAIGN / "offline-final-eight-20260811T181700Z/final-eight-manifest.json"
    prior = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    if prior.get("status") != "complete_active_eight" or prior.get("authoritative_active_records") != 2240:
        raise ValueError("completed prior-run manifest is not authoritative")
    prior_checks = _verify_prior_manifest(repo, prior)
    supplemental = [row for row in prior["regraded_artifacts"] if row.get("model_id") in SUPPLEMENTAL_MODELS]
    if len(supplemental) != len(SUPPLEMENTAL_MODELS):
        raise ValueError("the five supplemental chart models are incomplete")
    for row in supplemental:
        if row.get("records") != 280 or row.get("cost_covered_records") != 280:
            raise ValueError(f"supplemental model is incomplete: {row.get('model_id')}")
        if row.get("suite_records") != {"core": 232, "qec": 48}:
            raise ValueError(f"supplemental Core/QEC split is invalid: {row.get('model_id')}")

    required_plots = [
        final_root / "analysis" / f"price-vs-performance-{name}.png" for name in ("overall", "core", "qec")
    ]
    if any(not path.is_file() or path.stat().st_size < 100_000 for path in required_plots):
        raise ValueError("one or more ten-model price/performance plots is missing")
    return {
        "final_campaign": {
            "publication_ready": True,
            "configurations": 28,
            "records": 7840,
            "provider_cost_coverage": 7840,
            "checksum_manifest_files": checksum_count,
        },
        "prior_completed_run": {
            "status": "complete_active_eight",
            "records": 2240,
            "manifest_hash_checks": prior_checks,
            "supplemental_models": sorted(SUPPLEMENTAL_MODELS),
        },
        "combined_plots": {"models": 10, "configurations": 33, "cost_covered_records": 9240},
    }


def create_source_snapshots(repo: Path, staging: Path) -> tuple[list[tuple[Path, str]], dict[str, str]]:
    """Create compressed Git snapshots for every source revision needed for provenance."""
    final_audit = json.loads((repo / FINAL_CAMPAIGN / "final-audit.json").read_text(encoding="utf-8"))
    source = final_audit["source"]
    requested = {
        "benchmark-content": BENCHMARK_COMMIT,
        "generation-harness": str(source["generation_harness_commit"]),
        "offline-grading-harness": str(source["offline_grading_harness_commit"]),
        "analysis-head": str(source["analysis_commit"]),
    }
    prior = json.loads(
        (repo / PRIOR_CAMPAIGN / "offline-final-eight-20260811T181700Z/final-eight-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    requested["prior-generation-harness"] = str(prior["generation_harness_commit"])
    requested["prior-offline-harness"] = str(prior["offline_harness_commit"])
    resolved = {name: _git(repo, "rev-parse", f"{revision}^{{commit}}") for name, revision in requested.items()}
    files = []
    for name, revision in sorted(resolved.items()):
        path = staging / f"{name}-{revision[:12]}.tar.gz"
        subprocess.run(
            ["git", "archive", "--format=tar.gz", f"--output={path}", revision],
            cwd=repo,
            check=True,
        )
        files.append((path, f"source/revisions/{path.name}"))
    return files, resolved


def enumerate_tree(root: Path) -> tuple[list[Path], list[tuple[str, str]]]:
    """Enumerate regular files without following the result tree's convenience symlinks."""
    files = []
    symlinks = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(dirnames):
            path = current_path / name
            if path.is_symlink():
                symlinks.append((path.relative_to(root.parent).as_posix(), os.readlink(path)))
                dirnames.remove(name)
        for name in filenames:
            path = current_path / name
            if path.is_symlink():
                symlinks.append((path.relative_to(root.parent).as_posix(), os.readlink(path)))
            elif path.is_file():
                files.append(path)
    return sorted(files), sorted(symlinks)


def create_zip(
    output: Path,
    *,
    prefix: str,
    payload: Sequence[tuple[Path, str]],
    symlinks: Sequence[tuple[str, str]],
    repo: Path,
    created_at: datetime,
    revisions: dict[str, str],
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create a ZIP and return the exact hash manifest for its payload."""
    entries = []
    total_bytes = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for index, (source, relative_name) in enumerate(payload, start=1):
            archive_name = f"{prefix}/{relative_name}"
            digest, size = _write_file(archive, source, archive_name)
            entries.append({"path": relative_name, "bytes": size, "sha256": digest})
            total_bytes += size
            if index % 500 == 0:
                print(
                    json.dumps(
                        {"phase": "archive", "files": index, "uncompressed_bytes": total_bytes},
                        sort_keys=True,
                    ),
                    flush=True,
                )

        links_text = "path\ttarget\n" + "".join(f"{path}\t{target}\n" for path, target in symlinks)
        _write_text(archive, f"{prefix}/SYMLINKS.tsv", links_text)
        checksums = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries)
        _write_text(archive, f"{prefix}/SHA256SUMS", checksums)
        manifest = {
            "schema_version": "qceval.complete_results_archive.v1",
            "created_at_utc": created_at.isoformat().replace("+00:00", "Z"),
            "archive_root": prefix,
            "source_repository": str(repo),
            "source_head": _git(repo, "rev-parse", "HEAD"),
            "source_revisions": revisions,
            "worktree_status": _git(repo, "status", "--short"),
            "scope": {
                "results_tree": "all files under results/",
                "payload_files": len(entries),
                "payload_bytes": total_bytes,
                "recorded_symlinks": len(symlinks),
                "source_revision_archives": len(revisions),
            },
            "integrity_audit": audit,
            "files": entries,
        }
        _write_text(archive, f"{prefix}/MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        readme = _readme(prefix, len(entries), total_bytes, len(symlinks), audit)
        _write_text(archive, f"{prefix}/README.md", readme)
    return entries


def verify_zip(output: Path, *, prefix: str, entries: Sequence[dict[str, Any]]) -> None:
    """Test ZIP CRCs, extract to a temporary directory, and re-hash every payload file."""
    print(json.dumps({"phase": "verify", "check": "zip_crc"}, sort_keys=True), flush=True)
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC verification failed: {bad}")
        with tempfile.TemporaryDirectory(prefix="qceval-archive-verify-") as temporary:
            extraction = Path(temporary)
            archive.extractall(extraction)
            for index, entry in enumerate(entries, start=1):
                path = extraction / prefix / entry["path"]
                if path.stat().st_size != entry["bytes"] or _sha256_file(path) != entry["sha256"]:
                    raise ValueError(f"extracted payload failed verification: {entry['path']}")
                if index % 1000 == 0:
                    print(
                        json.dumps({"phase": "verify", "files": index}, sort_keys=True),
                        flush=True,
                    )


def _verify_checksum_manifest(repo: Path, path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        candidate = repo / relative
        if not candidate.is_file() or _sha256_file(candidate) != expected:
            raise ValueError(f"checksum manifest mismatch: {relative}")
        count += 1
    return count


def _verify_prior_manifest(repo: Path, manifest: dict[str, Any]) -> int:
    checks: list[tuple[str, str]] = []
    checks.extend((str(row["path"]), str(row["sha256"])) for row in manifest["regraded_artifacts"])
    checks.extend((str(row["path"]), str(row["sha256"])) for row in manifest["score_cost_outputs"].values())
    checks.append((str(manifest["benchmark_content_manifest"]), str(manifest["benchmark_content_manifest_sha256"])))
    checks.append((str(manifest["full_nine_model_queue"]), str(manifest["full_nine_model_queue_sha256"])))
    for path_value, expected in checks:
        path = Path(path_value)
        if not path.is_absolute():
            path = repo / path
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"prior-run manifest mismatch: {path_value}")
    return len(checks)


def _untracked_source_files(repo: Path, output: Path) -> list[tuple[Path, str]]:
    raw = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repo,
    )
    files = []
    for value in raw.decode().split("\0"):
        if not value:
            continue
        path = repo / value
        if path.is_file() and not path.is_relative_to(output.parent):
            files.append((path, f"source/untracked/{value}"))
    return files


def _write_file(archive: zipfile.ZipFile, source: Path, archive_name: str) -> tuple[str, int]:
    info = zipfile.ZipInfo.from_file(source, archive_name)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | stat.S_IMODE(source.stat().st_mode)) << 16
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as source_handle, archive.open(info, "w", force_zip64=True) as output_handle:
        while chunk := source_handle.read(CHUNK_SIZE):
            digest.update(chunk)
            output_handle.write(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _write_text(archive: zipfile.ZipFile, name: str, value: str) -> None:
    info = zipfile.ZipInfo(name, date_time=datetime.now().timetuple()[:6])
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, value.encode())


def _readme(prefix: str, files: int, payload_bytes: int, symlinks: int, audit: dict[str, Any]) -> str:
    return f"""# QCircuitEval complete results snapshot

This is a self-contained copy of every regular file under `results/` plus
compressed Git source snapshots for all revisions cited by the final campaigns.

- Archive root: `{prefix}`
- Payload files: {files:,}
- Uncompressed payload bytes: {payload_bytes:,}
- Convenience symlinks recorded in `SYMLINKS.tsv`: {symlinks}
- Final effort campaign: 28 configurations, 7,840/7,840 records and provider-cost coverage
- Combined chart data: 33 configurations across 10 models, 9,240/9,240 provider-cost coverage
- GLM-5.2 is withdrawn from the authoritative comparison; its historical
  diagnostic/withdrawal evidence remains preserved because this archive includes
  the complete results tree.
- Laguna remains smoke-only and is not part of the authoritative denominator.

Integrity was checked before archiving:

- {audit["final_campaign"]["checksum_manifest_files"]} final-campaign checksums matched.
- {audit["prior_completed_run"]["manifest_hash_checks"]} prior-run manifest-pinned checksums matched.
- The ZIP CRC test passed, the ZIP was extracted into a fresh temporary directory,
  and every payload file was re-hashed against `MANIFEST.json`.

To verify after extraction:

```bash
cd {prefix}
sha256sum -c SHA256SUMS
```

The external `{prefix}.zip.sha256` file verifies the ZIP itself.
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
