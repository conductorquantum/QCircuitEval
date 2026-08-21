"""Generated-code extraction from normalized Coda events."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from typing import Any

from qceval.core.code import extract_code_from_text
from qceval.models import Framework
from qceval.providers.coda.event_fields import (
    _event_type,
    _text_from_event,
)
from qceval.providers.coda.event_parsing import _maybe_json
from qceval.providers.coda.event_types import (
    PLUMBING_EVENT_TYPES,
    CodaCodeExtraction,
    CodaEventStream,
    _GeneratedCodeText,
)

_CIRCUIT_TOOL_NAMES = frozenset({"add_circuit_to_pipeline", "update_circuit_code"})


def extract_coda_generated_code(
    stream: CodaEventStream,
    *,
    entry_point: str,
    framework: Framework,
    prefer_structured_response: bool = False,
) -> CodaCodeExtraction:
    """Extract the most complete generated code already emitted by Coda.

    Args:
        stream: Parsed Coda event stream.
        entry_point: Required public function name in generated code.
        framework: Target quantum framework.
        prefer_structured_response: Whether complete structured responses
            should outrank complete unstructured emissions.

    Returns:
        Extracted code, source text, and emission-source metadata.
    """
    generated_texts = _generated_code_texts_from_stream(
        stream,
        framework,
        entry_point,
    )
    extraction_text = _most_complete_generated_code_text(
        generated_texts,
        entry_point,
        prefer_structured_response=prefer_structured_response,
    )
    if extraction_text is None:
        return CodaCodeExtraction(code=None, text=None, source=None)
    return CodaCodeExtraction(
        code=extract_code_from_text(
            extraction_text.text,
            entry_point,
            prefer_last=True,
        ),
        text=extraction_text.text,
        source=extraction_text.source,
    )


def _generated_code_texts_from_stream(
    stream: CodaEventStream,
    framework: Framework,
    entry_point: str,
) -> list[_GeneratedCodeText]:
    """Collect emitted text fields that may contain generated code."""
    generated_texts: list[_GeneratedCodeText] = []
    for text in stream.final_code_texts:
        _add_generated_code_text(
            generated_texts,
            text,
            "final_generated_code",
            structured=False,
        )
    for text in _circuit_tool_code(stream.events):
        _add_generated_code_text(
            generated_texts,
            text,
            "circuit_tool_code",
            structured=False,
        )
    _add_markdown_extracted_code_texts(
        generated_texts,
        stream.token_text,
        "token",
        entry_point,
    )
    for text in stream.message_texts:
        _add_markdown_extracted_code_texts(
            generated_texts,
            text,
            "message",
            entry_point,
        )
    for data in stream.structured_data:
        for text in _structured_texts(data, framework):
            _add_generated_code_text(
                generated_texts,
                text,
                "structured_response",
                structured=True,
            )
    for event in stream.events:
        _add_generated_code_text(
            generated_texts,
            _fallback_text(event),
            "fallback",
            structured=False,
        )
    return generated_texts


def _circuit_tool_code(
    events: tuple[dict[str, Any], ...],
) -> list[str]:
    """Extract code from circuit-tool calls in reverse chronological order."""
    codes: list[str] = []
    for event in events:
        if _event_type(event) != "tool_call":
            continue
        name = event.get("name", "")
        if name not in _CIRCUIT_TOOL_NAMES:
            continue
        code = _extract_code_from_event_args(event)
        if code:
            codes.append(code)
    codes.reverse()
    return codes


def _extract_code_from_event_args(event: Mapping[str, Any]) -> str:
    """Try supported paths to find code in a tool-call event."""
    for key in ("args", "arguments", "input", "data"):
        raw = event.get(key)
        if raw is None:
            continue
        if isinstance(raw, str):
            raw = _maybe_json(raw) or {}
        if isinstance(raw, Mapping):
            code = raw.get("code", "")
            if isinstance(code, str) and code.strip():
                return code
            nested = raw.get("arguments") or raw.get("input") or raw.get("args") or {}
            if isinstance(nested, str):
                nested = _maybe_json(nested) or {}
            if isinstance(nested, Mapping):
                code = nested.get("code", "")
                if isinstance(code, str) and code.strip():
                    return code
    return ""


def _add_generated_code_text(
    generated_texts: list[_GeneratedCodeText],
    text: str,
    source: str,
    *,
    structured: bool,
) -> None:
    if text.strip():
        generated_texts.append(
            _GeneratedCodeText(
                text=text,
                source=source,
                structured=structured,
                index=len(generated_texts),
            )
        )


def _add_markdown_extracted_code_texts(
    generated_texts: list[_GeneratedCodeText],
    text: str,
    source: str,
    entry_point: str,
) -> None:
    """Add markdown-stripped code before the raw emitted text."""
    if not text.strip():
        return
    extracted = extract_code_from_text(text, entry_point, prefer_last=True)
    if extracted != text.strip():
        _add_generated_code_text(
            generated_texts,
            extracted,
            f"{source}_extracted",
            structured=False,
        )
    _add_generated_code_text(
        generated_texts,
        text,
        source,
        structured=False,
    )


def _structured_texts(
    data: Mapping[str, Any],
    framework: Framework,
) -> list[str]:
    """Extract code strings from a structured response mapping."""
    texts: list[str] = []
    texts.extend(_code_strings(data.get(framework)))
    texts.extend(_code_strings(data.get("code")))
    for key, value in data.items():
        if key in {framework, "code"}:
            continue
        if _is_code_like_key(str(key)):
            texts.extend(_code_strings(value))
    return texts


def _code_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        fields = (
            value.get("code"),
            value.get("content"),
            value.get("output"),
        )
        return [item for item in fields if isinstance(item, str)]
    return []


def _is_code_like_key(key: str) -> bool:
    lowered = key.lower()
    return "code" in lowered or lowered in {
        "python",
        "qiskit",
        "cirq",
        "pennylane",
        "cudaq",
    }


def _fallback_text(event: Mapping[str, Any]) -> str:
    event_type = _event_type(event)
    excluded = {
        "token",
        "structured_response",
        "completed",
        "done",
        "error",
        "cancelled",
    }
    if event_type in PLUMBING_EVENT_TYPES | excluded:
        return ""
    if event_type in {"message", "assistant_message", "final_message"} or event.get("role") == "assistant":
        return ""
    return _text_from_event(event)


def _final_generated_code_text(event: Mapping[str, Any]) -> str:
    if _event_type(event) != "tool_result" or event.get("name") != "final_generated_code":
        return ""
    output = event.get("output")
    if isinstance(output, Mapping):
        code = output.get("code")
        return code if isinstance(code, str) else ""
    if isinstance(output, str):
        return _code_from_tool_result_repr(output)
    return ""


def _code_from_tool_result_repr(output: str) -> str:
    match = re.search(
        r"(?:^|\s)code=(?P<value>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')",
        output,
        re.DOTALL,
    )
    if match is not None:
        try:
            value = ast.literal_eval(match.group("value"))
            if isinstance(value, str) and not _is_incomplete_code(value):
                return value
        except (SyntaxError, ValueError):
            pass
    return ""


def _is_incomplete_code(code: str) -> bool:
    """Detect code ending with a bare Ellipsis placeholder."""
    stripped = code.rstrip()
    if stripped.endswith("..."):
        last_line = stripped.rsplit("\n", 1)[-1].strip()
        if last_line == "...":
            return True
    return False


def _most_complete_generated_code_text(
    generated_texts: Sequence[_GeneratedCodeText],
    entry_point: str,
    *,
    prefer_structured_response: bool,
) -> _GeneratedCodeText | None:
    """Return emitted text most likely to contain complete generated code."""
    confidence_pairs = [(text, _code_text_confidence(text.text, entry_point)) for text in generated_texts]
    confidence_pairs = [(text, confidence) for text, confidence in confidence_pairs if confidence > 0]
    if not confidence_pairs:
        return None
    return max(
        confidence_pairs,
        key=lambda item: _extraction_source_key(
            item[0],
            item[1],
            entry_point,
            prefer_structured_response,
        ),
    )[0]


def _extraction_source_key(
    generated_text: _GeneratedCodeText,
    confidence: int,
    entry_point: str,
    prefer_structured_response: bool,
) -> tuple[int, int, int]:
    """Build a comparison key ``(confidence, source_bonus, -index)``."""
    source_bonus = 0
    if prefer_structured_response and generated_text.structured and _has_entry_point(generated_text.text, entry_point):
        source_bonus = 2
    elif (
        not prefer_structured_response
        and not generated_text.structured
        and _has_entry_point(generated_text.text, entry_point)
    ):
        source_bonus = 1
    return confidence, source_bonus, -generated_text.index


def _code_text_confidence(text: str, entry_point: str) -> int:
    """Estimate whether emitted text contains extractable generated code."""
    has_ep = _has_entry_point(text, entry_point)
    if has_ep and not _has_markdown_fences(text):
        if _is_incomplete_code(text):
            return 2
        return 4
    if _python_block_with_entry_point(text, entry_point):
        return 3
    if has_ep:
        return 3
    if _has_python_block(text):
        return 2
    return 1 if text.strip() else 0


def _has_entry_point(text: str, entry_point: str) -> bool:
    return f"def {entry_point}" in text


def _has_markdown_fences(text: str) -> bool:
    return "```" in text


def _python_block_with_entry_point(text: str, entry_point: str) -> bool:
    return any(_has_entry_point(block, entry_point) for block in _python_blocks(text))


def _has_python_block(text: str) -> bool:
    return bool(_python_blocks(text))


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
