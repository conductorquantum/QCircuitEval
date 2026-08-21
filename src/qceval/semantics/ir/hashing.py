"""Content hashing for canonical Program IR."""

from __future__ import annotations

import hashlib
import json

from qceval.semantics.ir.canonicalize import canonical_program_dict
from qceval.semantics.ir.model import Program


def source_code_sha256(code: str | None) -> str:
    """Return the SHA-256 digest of candidate source text.

    Args:
        code: Candidate Python source, or ``None`` when unavailable.

    Returns:
        Lowercase hexadecimal digest of the UTF-8 source bytes.
    """
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()


def program_hash(program: Program) -> str:
    """Return the semantic SHA-256 digest of a Program IR.

    Args:
        program: Validated Program IR.

    Returns:
        Lowercase SHA-256 digest excluding provenance and diagnostics.
    """
    payload = json.dumps(
        canonical_program_dict(program),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()
