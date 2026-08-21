"""Build the wheel and prove packaged assets load from a clean install.

This catches editable-checkout-only asset paths that are missing from the
installed distribution. The wheel is built into an isolated temporary directory
and installed with ``--no-deps``. Parent packages that pull optional runtime
dependencies are stubbed so the smoke can exercise the same asset loaders used
in production without installing Cirq/Qiskit/CUDA-Q.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_SRC = REPO_ROOT / "src" / "qceval" / "assets"


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if completed.returncode == 0:
        return
    detail = (completed.stderr or completed.stdout or "").strip()
    raise SystemExit(f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}")


def _expected_wheel_assets() -> tuple[str, ...]:
    """Return wheel paths for every tracked file under ``src/qceval/assets``."""

    expected: list[str] = []
    for path in sorted(ASSETS_SRC.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(ASSETS_SRC).as_posix()
        expected.append(f"qceval/assets/{relative}")
    if not expected:
        raise SystemExit(f"no asset files found under {ASSETS_SRC}")
    return tuple(expected)


def _build_wheel(dist_dir: Path) -> Path:
    dist_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "hatchling",
            "build",
            "--target",
            "wheel",
            "--directory",
            str(dist_dir),
            "--clean",
        ],
        cwd=REPO_ROOT,
    )
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        names = ", ".join(path.name for path in wheels) or "<none>"
        raise SystemExit(f"expected exactly one wheel in {dist_dir}, found: {names}")
    return wheels[0]


def _assert_wheel_assets(wheel: Path, expected: tuple[str, ...]) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    missing = [name for name in expected if name not in names]
    if missing:
        raise SystemExit("wheel missing packaged assets:\n" + "\n".join(missing))


def _smoke_installed_wheel(wheel: Path, work_dir: Path) -> None:
    env_dir = work_dir / "venv"
    _run(["uv", "venv", str(env_dir)])
    bin_dir = env_dir / ("Scripts" if sys.platform == "win32" else "bin")
    python = bin_dir / "python"
    env = {**os.environ, "VIRTUAL_ENV": str(env_dir)}
    _run(["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)], env=env)
    script = r"""
from __future__ import annotations

import importlib
import sys
import sysconfig
import types
from pathlib import Path

root = Path(sysconfig.get_paths()["purelib"]) / "qceval"
assert root.is_dir(), root


def _shell(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__file__ = str(path / "__init__.py")
    module.__path__ = [str(path)]
    sys.modules[name] = module


# Stub only packages whose __init__ imports optional runtime dependencies.
_shell("qceval", root)
_shell("qceval.evals", root / "evals")
_shell("qceval.semantics", root / "semantics")
_shell("qceval.semantics.contracts", root / "semantics" / "contracts")
_shell("qceval.semantics.targets", root / "semantics" / "targets")

resources = importlib.import_module("qceval.assets._resources")
tasks = importlib.import_module("qceval.evals.tasks")
registry_mod = importlib.import_module("qceval.semantics.contracts.registry")
targets_load = importlib.import_module("qceval.semantics.targets.load")

assert resources.contract_resource("core").is_file()
assert resources.task_resource("qec", "qiskit").is_file()
assert resources.target_resource("core", "manifest.json").is_file()

core_tasks = tasks.load_tasks("qiskit", "core")
qec_tasks = tasks.load_tasks("qiskit", "qec")
core_registry = registry_mod.ContractRegistry.from_package("core")
qec_registry = registry_mod.ContractRegistry.from_package("qec")
assert len(core_tasks) == len(core_registry) > 0
assert len(qec_tasks) == len(qec_registry) > 0
core_doc = targets_load.load_contract_target_document(core_registry.get("core", "01"))
qec_doc = targets_load.load_contract_target_document(qec_registry.get("qec", "qec01"))
assert core_doc["task_id"] == "01", core_doc.get("task_id")
assert qec_doc["task_id"] == "qec01", qec_doc.get("task_id")
print("packaged asset wheel smoke ok")
"""
    _run([str(python), "-c", script], env=env)


def main() -> int:
    """Build, inventory, and smoke-test the installed wheel assets."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    expected = _expected_wheel_assets()
    with tempfile.TemporaryDirectory(prefix="qceval-wheel-assets-") as temporary:
        work_dir = Path(temporary)
        wheel = _build_wheel(work_dir / "dist")
        _assert_wheel_assets(wheel, expected)
        _smoke_installed_wheel(wheel, work_dir)
    print(f"packaged asset wheel check passed ({len(expected)} assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
