"""Regenerate or verify the packaged core semantic contract and target assets.

The declarative audit source (``core-audit-source.json``) plus the framework
task assets fully determine the packaged contracts and the suite-level
``targets/core/manifest.json`` and ``targets/core/target.json`` documents.
Run without arguments to rewrite the packaged files; run with ``--check`` to
fail when any packaged byte differs from its deterministic generator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from qceval.semantics.core_audit import generated_core_assets  # noqa: E402


def main() -> int:
    """Generate or verify every packaged semantic asset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify packaged bytes instead of rewriting")
    arguments = parser.parse_args()
    files = dict(generated_core_assets())
    if not arguments.check:
        for path, payload in sorted(files.items()):
            target = REPO_ROOT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        print(f"wrote {len(files)} semantic assets")
        return 0
    stale = [str(path) for path, payload in sorted(files.items()) if (REPO_ROOT / path).read_bytes() != payload]
    if stale:
        print("stale semantic assets:\n" + "\n".join(stale))
        return 1
    print(f"verified {len(files)} semantic assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
