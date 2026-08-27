"""Optional utility: export a 28-instance category sample of QCircuitEval.

The public dataset is the full 280-instance task set in ``src/qceval/assets``.
This script remains as a maintainer convenience: it deterministically writes
exactly one representative task per prompt category (7 tasks), for each of
the four supported frameworks (28 task instances total), into ``public_subset/``.

Usage:
    python3 ci/export_public_subset.py

Uses only the Python standard library and is safe to re-run (output files
are overwritten deterministically).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "src" / "qceval" / "assets"
OUTPUT_DIR = REPO_ROOT / "public_subset"

FRAMEWORKS = ("qiskit", "cirq", "pennylane", "cudaq")

# One representative task per prompt category.
# suite -> ordered list of (task_id, prompt_category, description)
PUBLIC_TASKS: dict[str, list[tuple[str, str, str]]] = {
    "core": [
        (
            "01",
            "Oracular / Query",
            "Grover search with a marked-state oracle.",
        ),
        (
            "08",
            "Algebraic / Fourier / Number-Theoretic",
            "Quantum Fourier transform on 6 qubits.",
        ),
        (
            "54",
            "Approximation / Simulation / Quantum Walks",
            "Second-order Trotterized Heisenberg model evolution.",
        ),
        (
            "04",
            "Optimization / Numerics / Estimation / Variational",
            "QAOA MaxCut ansatz circuit.",
        ),
        (
            "02",
            "State Preparation / Quantum Information",
            "Preparation of a specified 3-qubit quantum state.",
        ),
        (
            "21",
            "Circuit Synthesis / Reversible Logic / Decomposition",
            "3-bit majority (MAJ) reversible-logic circuit.",
        ),
    ],
    "qec": [
        (
            "qec01",
            "Quantum Error Correction",
            "Three-qubit bit-flip code encode/decode circuit.",
        ),
    ],
}


def load_tasks(suite: str, framework: str) -> dict[str, dict]:
    """Load a framework's JSONL asset file as a mapping of task_id -> task."""
    path = ASSETS_DIR / suite / f"{framework}.jsonl"
    tasks: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            tasks[task["task_id"]] = task
    return tasks


def main() -> None:
    manifest_categories = []
    total_instances = 0
    semantic_contracts = 0

    for suite, selections in PUBLIC_TASKS.items():
        out_dir = OUTPUT_DIR / suite
        out_dir.mkdir(parents=True, exist_ok=True)
        per_framework: dict[str, list[dict]] = {fw: [] for fw in FRAMEWORKS}

        for framework in FRAMEWORKS:
            tasks = load_tasks(suite, framework)
            for task_id, _category, _description in selections:
                if task_id not in tasks:
                    raise KeyError(f"Task {task_id!r} not found in {suite}/{framework}.jsonl")
                per_framework[framework].append(tasks[task_id])

        for framework in FRAMEWORKS:
            out_path = out_dir / f"{framework}.jsonl"
            with out_path.open("w", encoding="utf-8") as fh:
                for task in per_framework[framework]:
                    fh.write(json.dumps(task, sort_keys=True) + "\n")
            total_instances += len(per_framework[framework])

        semantic_contracts += export_semantic_artifacts(suite, selections, out_dir)

        reference = load_tasks(suite, FRAMEWORKS[0])
        for task_id, category, description in selections:
            manifest_categories.append(
                {
                    "category": category,
                    "suite": suite,
                    "task_id": task_id,
                    "entry_point": reference[task_id]["entry_point"],
                    "description": description,
                }
            )

    manifest = {
        "name": "QCircuitEval category sample",
        "description": (
            "Optional 28-instance category sample of QCircuitEval: one "
            "representative task per prompt category, across all four "
            "frameworks. The full 280-instance task set is public in "
            "src/qceval/assets."
        ),
        "frameworks": list(FRAMEWORKS),
        "num_tasks": len(manifest_categories),
        "num_frameworks": len(FRAMEWORKS),
        "num_task_instances": total_instances,
        "categories": manifest_categories,
        "semantic_artifacts": {
            "contract_schema_version": "1",
            "contracts": semantic_contracts,
            "score_authority": "behavior",
            "hidden_corpus_included": False,
        },
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(
        f"Exported {len(manifest_categories)} tasks x {len(FRAMEWORKS)} "
        f"frameworks = {total_instances} task instances to {OUTPUT_DIR}"
    )


def export_semantic_artifacts(suite: str, selections: list[tuple[str, str, str]], out_dir: Path) -> int:
    """Export selected contracts and independent targets.

    Args:
        suite: Benchmark suite.
        selections: Public task selections.
        out_dir: Suite-specific public output directory.

    Returns:
        Number of exported semantic contracts.
    """
    contract_path = ASSETS_DIR / "contracts" / f"{suite}.jsonl"
    if not contract_path.exists():
        return 0
    task_ids = {task_id for task_id, _, _ in selections}
    contracts = [
        line
        for line in contract_path.read_text(encoding="utf-8").splitlines()
        if line and str(json.loads(line)["task_id"]) in task_ids
    ]
    (out_dir / "semantic_contracts.jsonl").write_text("\n".join(contracts) + "\n", encoding="utf-8")

    manifest_doc = json.loads((ASSETS_DIR / "targets" / suite / "manifest.json").read_text(encoding="utf-8"))
    target_doc = json.loads((ASSETS_DIR / "targets" / suite / "target.json").read_text(encoding="utf-8"))
    selected_manifests = {task_id: manifest_doc["tasks"][task_id] for task_id in sorted(task_ids)}
    selected_targets = {task_id: target_doc["tasks"][task_id] for task_id in sorted(task_ids)}
    targets_dir = out_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    (targets_dir / "manifest.json").write_text(
        json.dumps(
            {"schema_version": "1", "suite": suite, "tasks": selected_manifests},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (targets_dir / "target.json").write_text(
        json.dumps(
            {"schema_version": "1", "suite": suite, "tasks": selected_targets},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(contracts)


if __name__ == "__main__":
    main()
