"""Tests for Coda event parsing and generated-code extraction."""

from __future__ import annotations

from qceval.providers.coda.events import compact_event, extract_coda_generated_code, parse_coda_events


def test_parse_coda_data_events_accumulates_tokens() -> None:
    lines = [
        b'data: {"type": "token", "content": "```python\\n"}\n',
        b'data: {"type": "token", "content": "def answer():\\n    return 1\\n```"}\n',
        b'data: {"type": "token", "content": "<DONE>"}\n',
        b'data: {"type": "completed"}\n',
    ]
    stream = parse_coda_events(lines)
    assert stream.token_text == "```python\ndef answer():\n    return 1\n```"
    assert stream.completed is True
    assert stream.event_types["token"] == 3


def test_parse_coda_named_event_with_json_data() -> None:
    lines = [
        "event: token\n",
        'data: {"content": "def answer():\\n    return 1"}\n',
        "\n",
    ]
    stream = parse_coda_events(lines)
    assert stream.events[0]["type"] == "token"
    assert stream.events[0]["name"] == "token"
    assert stream.token_text == "def answer():\n    return 1"


def test_parse_coda_named_event_with_plain_text_data() -> None:
    lines = ["event: final_message\n", "data: def answer():\n", "\n"]
    stream = parse_coda_events(lines)
    assert stream.events[0]["type"] == "final_message"
    assert stream.message_texts == ("def answer():",)


def test_parse_coda_ignores_keepalive_and_plumbing_for_text() -> None:
    lines = [
        ": keepalive\n",
        'data: {"type": "heartbeat", "content": "ignored"}\n',
        'data: {"type": "token", "data": {"content": "def answer():\\n    return 1"}}\n',
    ]
    stream = parse_coda_events(lines)
    assert stream.token_text == "def answer():\n    return 1"
    assert stream.event_types["heartbeat"] == 1


def test_parse_coda_decodes_nested_json_data() -> None:
    lines = ['data: {"type": "structured_response", "data": "{\\"code\\": \\"def answer():\\\\n    return 1\\"}"}\n']
    stream = parse_coda_events(lines)
    assert stream.structured_data[0]["code"] == "def answer():\n    return 1"


def test_parse_coda_plain_json_fallback() -> None:
    lines = ['{"type": "token", "content": "def answer():\\n    return 1"}']
    stream = parse_coda_events(lines)
    assert stream.token_text == "def answer():\n    return 1"


def test_parse_coda_json_fallback_list_and_events_key() -> None:
    list_lines = ['[{"event": "token", "data": {"message": "def answer():\\n    return 1"}}]']
    events_lines = ['{"events": [{"type": "token", "text": "def answer():\\n    return 2"}]}']
    list_stream = parse_coda_events(list_lines)
    events_stream = parse_coda_events(events_lines)
    assert list_stream.token_text == "def answer():\n    return 1"
    assert events_stream.token_text == "def answer():\n    return 2"


def test_parse_coda_done_event_is_success_terminal() -> None:
    lines = ["data: [DONE]\n"]
    stream = parse_coda_events(lines)
    assert stream.completed is True
    assert stream.event_types["done"] == 1


def test_coda_extraction_prefers_token_source_by_default() -> None:
    lines = [
        'data: {"type": "token", "content": "def answer():\\n    return 1"}\n',
        (
            'data: {"type": "structured_response", "data": '
            '{"qiskit": "def answer():\\n    return 2", "code": "def answer():\\n    return 3"}}\n'
        ),
    ]
    stream = parse_coda_events(lines)
    extraction = extract_coda_generated_code(stream, entry_point="answer", framework="qiskit")
    assert extraction.source == "token"
    assert extraction.code == "def answer():\n    return 1"


def test_coda_extraction_prefers_structured_source_when_requested() -> None:
    lines = [
        'data: {"type": "token", "content": "def answer():\\n    return 1"}\n',
        'data: {"type": "structured_response", "data": {"qiskit": "def answer():\\n    return 2"}}\n',
    ]
    stream = parse_coda_events(lines)
    extraction = extract_coda_generated_code(
        stream,
        entry_point="answer",
        framework="qiskit",
        prefer_structured_response=True,
    )
    assert extraction.source == "structured_response"
    assert extraction.code == "def answer():\n    return 2"


def test_coda_extraction_prefers_final_generated_code_tool_result() -> None:
    lines = [
        'data: {"type": "token", "content": "```python\\ndef answer():\\n    return 1\\n```"}\n',
        (
            'data: {"type": "tool_result", "name": "final_generated_code", '
            "\"output\": \"circuit_name='answer' code='def answer():\\\\n    return 2'\"}\n"
        ),
    ]
    stream = parse_coda_events(lines)
    extraction = extract_coda_generated_code(stream, entry_point="answer", framework="qiskit")
    assert extraction.source == "final_generated_code"
    assert extraction.code == "def answer():\n    return 2"


def test_coda_extraction_uses_latest_matching_block() -> None:
    lines = [
        (
            'data: {"type": "token", "content": '
            '"```python\\ndef answer():\\n    return 1\\n```\\n```python\\ndef answer():\\n    return 2\\n```"}\n'
        )
    ]
    stream = parse_coda_events(lines)
    extraction = extract_coda_generated_code(stream, entry_point="answer", framework="qiskit")
    assert extraction.code == "def answer():\n    return 2"


def test_coda_extraction_accepts_structured_string_data() -> None:
    stream = parse_coda_events(['data: {"type": "structured_response", "data": "def answer():\\n    return 1"}\n'])
    extraction = extract_coda_generated_code(stream, entry_point="answer", framework="qiskit")
    assert extraction.source == "structured_response"
    assert extraction.code == "def answer():\n    return 1"


def test_coda_extraction_uses_code_like_structured_fields() -> None:
    stream = parse_coda_events(
        [
            (
                'data: {"type": "structured_response", "data": '
                '{"other_code": {"output": "def answer():\\n    return 1"}}}\n'
            )
        ]
    )
    extraction = extract_coda_generated_code(stream, entry_point="answer", framework="qiskit")
    assert extraction.source == "structured_response"
    assert extraction.code == "def answer():\n    return 1"


def test_coda_extraction_uses_unknown_event_as_fallback() -> None:
    stream = parse_coda_events(['data: {"type": "final", "output": "def answer():\\n    return 1"}\n'])
    extraction = extract_coda_generated_code(stream, entry_point="answer", framework="qiskit")
    assert extraction.source == "fallback"
    assert extraction.code == "def answer():\n    return 1"


def test_coda_extraction_accepts_python_block_without_entry_point() -> None:
    stream = parse_coda_events(['data: {"type": "token", "content": "```python\\nprint(1)\\n```"}\n'])
    extraction = extract_coda_generated_code(stream, entry_point="answer", framework="qiskit")
    assert extraction.code == "print(1)"


def test_coda_extraction_extracts_code_from_prose_with_blocks() -> None:
    lines = [
        (
            'data: {"type": "token", "content": '
            '"Here is the code:\\n\\n```python\\ndef answer():\\n    return 42\\n```\\n\\nThis returns 42."}\n'
        )
    ]
    stream = parse_coda_events(lines)
    extraction = extract_coda_generated_code(stream, entry_point="answer", framework="qiskit")
    assert extraction.source == "token_extracted"
    assert extraction.code == "def answer():\n    return 42"


def test_coda_extraction_returns_none_without_usable_text() -> None:
    stream = parse_coda_events(['data: {"type": "heartbeat", "content": "ignored"}\n'])
    extraction = extract_coda_generated_code(stream, entry_point="answer", framework="qiskit")
    assert extraction.code is None
    assert extraction.source is None


def test_coda_terminal_error_and_cancelled_are_recorded() -> None:
    error_stream = parse_coda_events(['data: {"type": "error", "content": "boom"}\n'])
    cancelled_stream = parse_coda_events(['data: {"type": "cancelled"}\n'])
    assert error_stream.terminal_error == "boom"
    assert cancelled_stream.cancelled is True


def test_coda_compact_event_truncates_large_strings() -> None:
    event = {"type": "token", "content": "x" * 20, "data": {"output": "y" * 20}}
    compact = compact_event(event, max_string_chars=5)
    assert compact["content"] == "xxxxx...[truncated 15 chars]"
    assert compact["data"]["output"] == "yyyyy...[truncated 15 chars]"


def test_coda_compact_events_caps_large_streams() -> None:
    lines = [f'data: {{"type": "heartbeat", "content": "{index}"}}\n' for index in range(105)]
    stream = parse_coda_events(lines)
    assert len(stream.compact_events) == 100
    assert stream.compact_events[50]["type"] == "truncated"


def test_coda_compact_event_preserves_json_scalars_and_repr() -> None:
    event = {"type": "custom", "data": {"items": [1, True, None], "object": object()}}
    compact = compact_event(event)
    assert compact["data"]["items"] == [1, True, None]
    assert "object at" in compact["data"]["object"]
