from __future__ import annotations

import json
from pathlib import Path

from qceval.core.code import extract_code_from_text
from qceval.core.io import infer_format, write_output


def test_extract_code_prefers_block_with_entry_point() -> None:
    # Arrange
    text = "```python\ndef other():\n    pass\n```\n```python\ndef answer():\n    return 1\n```"

    # Act
    code = extract_code_from_text(text, "answer")

    # Assert
    assert code == "def answer():\n    return 1"


def test_extract_code_prefers_last_matching_block_by_default() -> None:
    # A draft followed by a correction grades on the correction, and every
    # provider selects the same block for the same response (audit L8).
    text = "```python\ndef answer():\n    return 1\n```\n```python\ndef answer():\n    return 2\n```"

    # Act
    code = extract_code_from_text(text, "answer")

    # Assert
    assert code == "def answer():\n    return 2"


def test_extract_code_can_prefer_first_matching_block() -> None:
    # Arrange
    text = "```python\ndef answer():\n    return 1\n```\n```python\ndef answer():\n    return 2\n```"

    # Act
    code = extract_code_from_text(text, "answer", prefer_last=False)

    # Assert
    assert code == "def answer():\n    return 1"


def test_extract_code_does_not_match_prefixed_entry_point_names() -> None:
    # Regression (audit L8): "def answer" must not match "def answer_helper".
    text = "```python\ndef answer_helper():\n    return 1\n```\n```python\ndef answer():\n    return 2\n```"

    # Act
    code = extract_code_from_text(text, "answer")

    # Assert
    assert code == "def answer():\n    return 2"


def test_extract_code_can_prefer_last_matching_block() -> None:
    # Arrange
    text = "```python\ndef answer():\n    return 1\n```\n```python\ndef answer():\n    return 2\n```"

    # Act
    code = extract_code_from_text(text, "answer", prefer_last=True)

    # Assert
    assert code == "def answer():\n    return 2"


def test_extract_code_returns_first_generic_block() -> None:
    # Arrange
    text = "```\ndef other():\n    pass\n```"

    # Act
    code = extract_code_from_text(text, "answer")

    # Assert
    assert code == "def other():\n    pass"


def test_extract_code_returns_plain_text() -> None:
    # Arrange
    text = "def answer():\n    return 1"

    # Act
    code = extract_code_from_text(text, "answer")

    # Assert
    assert code == text


def test_write_output_writes_json(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "results.json"
    payload = {
        "results": [],
        "summary": {"passed": 0},
        "schema_version": "v",
        "provider": "p",
        "model": None,
        "configuration_id": "model__effort-max",
        "qceval": {},
    }

    # Act
    write_output(path, payload)

    # Assert
    assert json.loads(path.read_text(encoding="utf-8"))["summary"]["passed"] == 0


def test_write_output_writes_jsonl(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "results.jsonl"
    payload = {
        "schema_version": "v",
        "provider": "p",
        "model": None,
        "configuration_id": "model__effort-max",
        "qceval": {},
        "results": [{"task_id": "01"}],
        "summary": {"passed": 1},
    }

    # Act
    write_output(path, payload)

    # Assert
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["kind"] == "result"
    summary = json.loads(lines[1])
    assert summary["kind"] == "summary"
    assert summary["configuration_id"] == "model__effort-max"


def test_infer_format_respects_explicit_json() -> None:
    # Arrange
    path = Path("results.jsonl")

    # Act
    output_format = infer_format(path, "json")

    # Assert
    assert output_format == "json"
