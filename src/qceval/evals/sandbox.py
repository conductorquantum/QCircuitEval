"""Minimal candidate-code execution sandbox.

The sandbox isolates candidate imports and relative file access in a temporary
working directory, but it does not provide security isolation.  Use worker
processes and timeouts for operational containment when running untrusted code.
"""

from __future__ import annotations

import os
import re
import tempfile
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_UNICODE_OPERATOR_TRANSLATION = str.maketrans({"\N{MINUS SIGN}": "-"})


@contextmanager
def ignore_candidate_library_warnings() -> Iterator[None]:
    """Ignore candidate/library deprecation and syntax warnings during grading.

    Pytest and ``PYTHONWARNINGS=error`` turn these into exceptions. Candidate
    source and framework libraries (for example deprecated Qiskit ``QFT``)
    must not become ``execution_error`` when the program is otherwise valid.

    Returns:
        A context manager that restores the prior warning filters on exit.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        yield


def _compiles(code: str) -> bool:
    try:
        with ignore_candidate_library_warnings():
            compile(code, "<test>", "exec")
        return True
    except SyntaxError:
        return False


def _extract_code_blocks(text: str) -> str | None:
    """Extract Python code from markdown fenced code blocks.

    When a provider returns prose plus code blocks, this keeps evaluation on the
    generated code instead of failing on wrapper text.
    """
    blocks = _CODE_BLOCK_RE.findall(text)
    if not blocks:
        return None
    combined = "\n\n".join(block.strip() for block in blocks)
    if _compiles(combined):
        return combined
    for block in reversed(blocks):
        candidate = block.strip()
        if _compiles(candidate):
            return candidate
    return None


def _sanitize_code(code: str) -> str:
    """Normalize provider output before execution.

    Fixes transport-level artifacts that do not alter candidate semantics:
    markdown fences, Unicode minus signs, and non-ASCII text inside
    comments/docstrings. Other executable source is preserved.
    """
    if _compiles(code):
        return code

    candidate = code.translate(_UNICODE_OPERATOR_TRANSLATION)
    if _compiles(candidate):
        return candidate

    extracted = _extract_code_blocks(candidate)
    if extracted is not None:
        return extracted

    return _strip_non_ascii_comments_and_docstrings(candidate)


def _strip_non_ascii_comments_and_docstrings(code: str) -> str:
    lines = code.split("\n")
    result = "\n".join(_clean_comment_docstring_lines(lines))
    if _compiles(result):
        return result
    return code


def _clean_comment_docstring_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    in_docstring = False
    docstring_char: str | None = None
    for line in lines:
        stripped = line.lstrip()
        if in_docstring:
            cleaned.append(_strip_non_ascii(line))
            if docstring_char and docstring_char in stripped:
                in_docstring = False
            continue

        docstring_char = _docstring_delimiter(stripped)
        if docstring_char is not None:
            cleaned.append(_strip_non_ascii(line))
            in_docstring = stripped.count(docstring_char) < 2
        elif stripped.startswith("#"):
            cleaned.append(_strip_non_ascii(line))
        else:
            cleaned.append(line)
    return cleaned


def _docstring_delimiter(stripped_line: str) -> str | None:
    if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
        return stripped_line[:3]
    return None


def _strip_non_ascii(line: str) -> str:
    return line.encode("ascii", "ignore").decode("ascii")


def execute_with_entry_point(code: str, entry_point: str, callback: Callable[[Any], Any]) -> Any:
    """Compile candidate source and run ``callback`` with its entry point.

    The callback runs while the temporary source file still exists.  Some
    frameworks inspect source locations lazily when decorated callables execute.

    Args:
        code: Candidate Python source.
        entry_point: Function name expected in ``code``.
        callback: Function called with the resolved entry point.

    Returns:
        Return value produced by ``callback``.

    Raises:
        RuntimeError: If ``entry_point`` is missing or not callable.
        SyntaxError: If sanitized ``code`` does not compile.
        Exception: Any exception raised by candidate code or ``callback``.
    """
    code = _sanitize_code(code)
    old_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix="qceval_task_") as workdir:
            workdir_path = Path(workdir)
            source_path = workdir_path / "candidate.py"
            source_path.write_text(code, encoding="utf-8")
            namespace: dict[str, Any] = {"__file__": str(source_path), "__name__": "__qceval_candidate__"}
            os.chdir(workdir_path)
            with ignore_candidate_library_warnings():
                exec(compile(code, str(source_path), "exec"), namespace, namespace)
                function = namespace.get(entry_point)
                if not callable(function):
                    raise RuntimeError(f"Entry point '{entry_point}' not found or not callable.")
                return callback(function)
    finally:
        os.chdir(old_cwd)


def execute_code_with_args(code: str, entry_point: str, *args: Any) -> Any:
    """Compile candidate source and call its entry point in a temp directory.

    Args:
        code: Candidate Python source.
        entry_point: Function name expected in ``code``.
        *args: Positional arguments passed to the entry point.

    Returns:
        Return value from the candidate entry point.

    Raises:
        RuntimeError: If ``entry_point`` is missing or not callable.
        SyntaxError: If ``code`` does not compile.
        Exception: Any exception raised by candidate code.
    """
    return execute_with_entry_point(code, entry_point, lambda function: function(*args))


def get_handler(task_id: str, code: str, entry_point: str, inputs: dict[str, Any]) -> Any:
    """Execute a candidate entry point with arity-derived task inputs.

    When an explicit ``call_args`` tuple is unavailable, bind arguments from the
    entry-point source arity and ``inputs[task_id]``. Contract-driven callers
    should prefer ``call_args_from_signature`` instead.

    Args:
        task_id: Zero-padded task identifier.
        code: Candidate Python source.
        entry_point: Function name to call.
        inputs: Deterministic task inputs keyed by task id.

    Returns:
        Candidate entry-point return value.

    Raises:
        RuntimeError: If ``entry_point`` is missing or not callable.
        SyntaxError: If ``code`` does not compile.
        Exception: Any exception raised by candidate code.
    """
    from qceval.semantics.contracts.binding import call_args_from_code

    sanitized = _sanitize_code(code)
    input_value = inputs.get(task_id)
    call_args = call_args_from_code(sanitized, entry_point, input_value)
    return execute_code_with_args(code, entry_point, *call_args)
