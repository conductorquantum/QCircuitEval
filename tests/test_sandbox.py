"""Tests for candidate-code sandbox execution and sanitization."""

from __future__ import annotations

import pytest

from qceval.evals.sandbox import execute_code_with_args, get_handler


def test_sandbox_executes_task_argument_handlers() -> None:
    code = "def answer(a, b=None, c=None):\n    return len([x for x in (a, b, c) if x is not None])\n"
    # Required arity is 1 (only ``a``), so the bundled input is passed as one arg.
    assert get_handler("06", code, "answer", {"06": "x"}) == 1
    spread = "def answer(a, b, c):\n    return a + b + c\n"
    assert get_handler("04", spread, "answer", {"04": [1, 2, 3]}) == 6
    assert get_handler("01", "def answer():\n    return 0\n", "answer", {}) == 0


def test_sandbox_reports_missing_entry_point() -> None:
    code = "def other():\n    return 1\n"
    with pytest.raises(RuntimeError) as exc:
        execute_code_with_args(code, "answer")
    assert "Entry point 'answer' not found" in str(exc.value)


def test_sandbox_preserves_source_for_inspection() -> None:
    code = (
        "import inspect\n"
        "def source_checked(function):\n"
        "    inspect.getsourcelines(function)\n"
        "    return function\n"
        "def answer():\n"
        "    @source_checked\n"
        "    def inner():\n"
        "        return 7\n"
        "    return inner()\n"
    )
    result = execute_code_with_args(code, "answer")
    assert result == 7


def test_sandbox_extracts_python_from_markdown_response() -> None:
    code = "Here is code:\n```python\ndef answer():\n    return 7\n```\n"
    result = execute_code_with_args(code, "answer")
    assert result == 7


def test_sandbox_normalizes_unicode_minus() -> None:
    minus = "\N{MINUS SIGN}"
    code = f"def answer():\n    return 3 {minus} 1\n"
    result = execute_code_with_args(code, "answer")
    assert result == 2


def test_sandbox_strips_non_ascii_docstring_text() -> None:
    code = 'def answer():\n    """ψ state note"""\n    return 3\n'
    result = execute_code_with_args(code, "answer")
    assert result == 3


def test_sandbox_uses_last_compiling_markdown_block() -> None:
    code = "```python\ndef answer(:\n    return 1\n```\n```python\ndef answer():\n    return 9\n```\n"
    result = execute_code_with_args(code, "answer")
    assert result == 9


def test_sandbox_raises_original_syntax_error_when_cleanup_cannot_repair() -> None:
    minus = "\N{MINUS SIGN}"
    code = f"def answer():\n    return (3 {minus}\n"
    with pytest.raises(SyntaxError):
        execute_code_with_args(code, "answer")
