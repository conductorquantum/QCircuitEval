"""Tests for Coda provider code extraction helpers."""

from __future__ import annotations

from qceval.providers.coda.event_extraction import (
    _code_from_tool_result_repr,
    _code_text_confidence,
    _extract_code_from_event_args,
    _is_incomplete_code,
)


class TestIsIncompleteCode:
    def test_complete_code(self) -> None:
        code = "def foo():\n    return 42\n"
        assert _is_incomplete_code(code) is False

    def test_ellipsis_on_own_line(self) -> None:
        code = "def foo():\n    x = 1\n    ..."
        assert _is_incomplete_code(code) is True

    def test_ellipsis_with_trailing_whitespace(self) -> None:
        code = "def foo():\n    x = 1\n    ...   \n"
        assert _is_incomplete_code(code) is True

    def test_ellipsis_in_string_literal(self) -> None:
        code = 'def foo():\n    return "hello..."\n'
        assert _is_incomplete_code(code) is False

    def test_ellipsis_as_pass_equivalent(self) -> None:
        code = "def foo():\n    x = [1, 2, ...]\n    return x"
        assert _is_incomplete_code(code) is False

    def test_empty_code(self) -> None:
        assert _is_incomplete_code("") is False

    def test_just_ellipsis(self) -> None:
        assert _is_incomplete_code("...") is True


class TestCodeFromToolResultRepr:
    def test_simple_single_quoted(self) -> None:
        output = "circuit_name='bell' code='from qiskit import QuantumCircuit\\nqc = QuantumCircuit(2)'"
        result = _code_from_tool_result_repr(output)
        assert "from qiskit import QuantumCircuit" in result
        assert "qc = QuantumCircuit(2)" in result

    def test_double_quoted(self) -> None:
        output = 'circuit_name="bell" code="def bell():\\n    return 1"'
        result = _code_from_tool_result_repr(output)
        assert "def bell():" in result

    def test_truncated_code_rejected(self) -> None:
        output = "circuit_name='x' code='def foo():\\n    x = 1\\n    ...'"
        result = _code_from_tool_result_repr(output)
        assert result == ""

    def test_no_code_field(self) -> None:
        output = "circuit_name='bell' something_else='value'"
        result = _code_from_tool_result_repr(output)
        assert result == ""

    def test_empty_code(self) -> None:
        output = "code=''"
        result = _code_from_tool_result_repr(output)
        assert result == ""


class TestExtractCodeFromEventArgs:
    def test_args_key_with_code(self) -> None:
        event = {"type": "tool_call", "name": "add_circuit_to_pipeline", "args": {"code": "def foo(): pass"}}
        assert _extract_code_from_event_args(event) == "def foo(): pass"

    def test_arguments_key(self) -> None:
        event = {"type": "tool_call", "arguments": {"code": "import qiskit"}}
        assert _extract_code_from_event_args(event) == "import qiskit"

    def test_input_key(self) -> None:
        event = {"type": "tool_call", "input": {"code": "x = 1"}}
        assert _extract_code_from_event_args(event) == "x = 1"

    def test_json_string_arguments(self) -> None:
        import json

        event = {"type": "tool_call", "args": json.dumps({"code": "return 42"})}
        assert _extract_code_from_event_args(event) == "return 42"

    def test_nested_arguments(self) -> None:
        event = {"type": "tool_call", "data": {"arguments": {"code": "nested code"}}}
        assert _extract_code_from_event_args(event) == "nested code"

    def test_no_code_field(self) -> None:
        event = {"type": "tool_call", "args": {"name": "bell"}}
        assert _extract_code_from_event_args(event) == ""

    def test_empty_event(self) -> None:
        event = {"type": "tool_call"}
        assert _extract_code_from_event_args(event) == ""

    def test_whitespace_only_code_ignored(self) -> None:
        event = {"type": "tool_call", "args": {"code": "   \n  "}}
        assert _extract_code_from_event_args(event) == ""


class TestCodeTextConfidence:
    def test_raw_code_with_entry_point_confidence_4(self) -> None:
        text = "def grover_search():\n    return qc\n"
        assert _code_text_confidence(text, "grover_search") == 4

    def test_incomplete_code_with_entry_point_confidence_2(self) -> None:
        text = "def grover_search():\n    x = 1\n    ..."
        assert _code_text_confidence(text, "grover_search") == 2

    def test_no_entry_point_has_low_confidence(self) -> None:
        text = "def other_function():\n    return 1\n"
        assert _code_text_confidence(text, "grover_search") <= 2

    def test_empty_text_confidence_0(self) -> None:
        assert _code_text_confidence("", "grover_search") == 0

    def test_markdown_with_entry_point_confidence_3(self) -> None:
        text = "```python\ndef grover_search():\n    return qc\n```"
        assert _code_text_confidence(text, "grover_search") == 3
