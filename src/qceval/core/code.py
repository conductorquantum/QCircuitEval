"""Helpers for extracting provider-generated Python source."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class _CodeBlock:
    """Single fenced code block extracted from Markdown text.

    Attributes:
        language: Lower-cased language tag (e.g. ``"python"``).
        code: Raw content between the opening and closing fences.
    """

    language: str
    code: str


def extract_code_from_text(text: str, entry_point: str, *, prefer_last: bool = True) -> str:
    """Extract candidate Python code from provider text.

    Providers often return Markdown instead of raw source.  This function first
    looks for fenced code blocks that define the requested entry point, then
    falls back to a fenced block, and finally to stripped raw text.

    Args:
        text: Provider response content.
        entry_point: Function name expected by the task.
        prefer_last: When true (the default), prefer the last matching block —
            a response showing a draft and then a correction is graded on the
            correction, and every provider grades the same response the same
            way.

    Returns:
        Candidate source code string.
    """
    blocks = _code_blocks(text)
    python_blocks = [block.code for block in blocks if _is_python_block(block)]
    generic_blocks = [block.code for block in blocks if not _is_python_block(block)]
    for group in (python_blocks, generic_blocks):
        selected = _matching_block(group, entry_point, prefer_last=prefer_last)
        if selected is not None:
            return selected.strip()
    if python_blocks:
        return _select_block(python_blocks, prefer_last=prefer_last).strip()
    if generic_blocks:
        return _select_block(generic_blocks, prefer_last=prefer_last).strip()
    return text.strip()


def _code_blocks(text: str) -> list[_CodeBlock]:
    pattern = re.compile(r"```(?P<language>[^\n`]*)\n?(?P<code>.*?)```", re.DOTALL)
    return [
        _CodeBlock(language=match.group("language").strip().lower(), code=match.group("code"))
        for match in pattern.finditer(text)
    ]


def _is_python_block(block: _CodeBlock) -> bool:
    return block.language in {"python", "py"}


def _matching_block(blocks: list[str], entry_point: str, *, prefer_last: bool) -> str | None:
    # Word-bounded so "def answer" does not match "def answer_helper".
    definition = re.compile(rf"\bdef\s+{re.escape(entry_point)}\s*\(")
    matches = [block for block in blocks if definition.search(block)]
    if not matches:
        return None
    return _select_block(matches, prefer_last=prefer_last)


def _select_block(blocks: list[str], *, prefer_last: bool) -> str:
    return blocks[-1] if prefer_last else blocks[0]
