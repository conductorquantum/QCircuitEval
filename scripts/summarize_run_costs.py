#!/usr/bin/env python3
"""Show benchmark score versus observed and extrapolated OpenRouter cost."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--lite-prompts", type=int, required=True)
    parser.add_argument("--target-prompts", type=int, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--tsv-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(argv)
    rows = summarize_costs(args.inputs, args.lite_prompts, args.target_prompts)
    if args.json_out is not None:
        _write(args.json_out, json.dumps(rows, indent=2, sort_keys=True) + "\n")
    if args.tsv_out is not None:
        _write(args.tsv_out, render_tsv(rows))
    if args.markdown_out is not None:
        _write(args.markdown_out, render_markdown(rows))
    print(render_tsv(rows), end="")
    return 0


def summarize_costs(inputs: Sequence[Path], lite_prompts: int, target_prompts: int) -> list[dict[str, Any]]:
    if lite_prompts < 1 or target_prompts < 1:
        raise ValueError("prompt counts must be positive")
    scale = target_prompts / lite_prompts
    rows = []
    for path in inputs:
        records, summary = _read_run(path)
        observed_cost = 0.0
        missing_cost_records = 0
        reported_cost_records = 0
        prompt_tokens = completion_tokens = reasoning_tokens = 0
        for record in records:
            response = record.get("provider_response") or {}
            usage = response.get("usage") or {}
            cost = _reported_cost(response)
            if cost is None:
                missing_cost_records += 1
            else:
                reported_cost_records += 1
                observed_cost += cost
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            reasoning_tokens += int(usage.get("reasoning_tokens") or 0)
        run_summary = summary["summary"]
        protocol = _protocol_name(run_summary["run_protocol"])
        score_name, score = _score(run_summary, protocol)
        logical_tasks = int(run_summary["task_totals"]["unique_tasks"])
        complete_cost = bool(records) and not missing_cost_records
        average_cost = observed_cost / logical_tasks if complete_cost and logical_tasks else None
        rows.append(
            {
                "model": summary.get("model"),
                "protocol": protocol,
                "score_name": score_name,
                "score": score,
                "lite_prompts_per_framework": lite_prompts,
                "target_prompts_per_framework": target_prompts,
                "logical_tasks": logical_tasks,
                "provider_records": len(records),
                "records_with_reported_cost": reported_cost_records,
                "records_missing_reported_cost": missing_cost_records,
                "reported_cost_coverage": reported_cost_records / len(records) if records else 0.0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "observed_cost_usd": observed_cost,
                "average_cost_per_task_usd": average_cost,
                "estimated_target_cost_usd": observed_cost * scale if complete_cost else None,
                "scale_factor": scale,
                "source": str(path),
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    totals: dict[str, float | None] = {}
    for model, model_rows in grouped.items():
        estimates = [row["estimated_target_cost_usd"] for row in model_rows]
        totals[model] = (
            sum(float(value) for value in estimates if value is not None)
            if all(value is not None for value in estimates)
            else None
        )
    for row in rows:
        row["estimated_all_protocols_cost_usd"] = totals[str(row["model"])]
    return sorted(rows, key=lambda row: (str(row["model"]), str(row["protocol"])))


def _read_run(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    summary = None
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload.get("kind") == "result":
            records.append(payload)
        elif payload.get("kind") == "summary":
            summary = payload
    if summary is None:
        raise ValueError(f"run has no summary: {path}")
    return records, summary


def _protocol_name(protocol: dict[str, Any]) -> str:
    if int(protocol["max_attempts"]) > 1:
        return f"feedback{int(protocol['max_attempts'])}"
    if int(protocol["samples_per_task"]) > 1:
        return f"pass{int(protocol['pass_k'])}"
    return "pass1"


def _reported_cost(response: Mapping[str, Any]) -> float | None:
    usage = response.get("usage")
    value = usage.get("cost_usd") if isinstance(usage, Mapping) else None
    if value is None:
        raw_response = response.get("raw_response")
        raw_usage = raw_response.get("usage") if isinstance(raw_response, Mapping) else None
        value = raw_usage.get("cost") if isinstance(raw_usage, Mapping) else None
    if value is None or isinstance(value, bool):
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    return cost if math.isfinite(cost) and cost >= 0.0 else None


def _score(summary: Mapping[str, Any], protocol: str) -> tuple[str, float | None]:
    if protocol.startswith("feedback"):
        lineage = summary.get("feedback_lineage")
        if isinstance(lineage, Mapping):
            return "feedback_terminal_pass_rate", _finite_rate(lineage.get("terminal_pass_rate"))
        feedback = summary.get("feedback")
        value = feedback.get("final_pass_rate") if isinstance(feedback, Mapping) else None
        return "feedback_final_pass_rate", _finite_rate(value)
    if protocol.startswith("pass") and protocol != "pass1":
        pass_at_k = summary.get("pass_at_k")
        value = pass_at_k.get("pass_at_k") if isinstance(pass_at_k, Mapping) else None
        return f"pass_at_{protocol[4:]}", _finite_rate(value)
    return "pass_rate", _finite_rate(summary.get("pass_rate"))


def _finite_rate(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    return rate if math.isfinite(rate) else None


def render_tsv(rows: Sequence[dict[str, Any]]) -> str:
    """Render the score-versus-average-cost deliverable as TSV."""
    lines = [
        "model\tprotocol\tscore_name\tscore\tavg_cost_per_task_usd\tobserved_usd\testimated_target_usd\tcost_coverage"
    ]
    for row in rows:
        lines.append(
            f"{row['model']}\t{row['protocol']}\t{row['score_name']}\t{_format_optional(row['score'])}\t"
            f"{_format_optional(row['average_cost_per_task_usd'], digits=8)}\t"
            f"{row['observed_cost_usd']:.6f}\t{_format_optional(row['estimated_target_cost_usd'])}\t"
            f"{row['records_with_reported_cost']}/{row['provider_records']}"
        )
    return "\n".join(lines) + "\n"


def render_markdown(rows: Sequence[dict[str, Any]]) -> str:
    """Render the score-versus-average-cost deliverable as Markdown."""
    lines = [
        "| Model | Protocol | Score | Average USD/task | Reported USD | Cost coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['protocol']} | {_format_optional(row['score'])} | "
            f"{_format_optional(row['average_cost_per_task_usd'], digits=8)} | "
            f"{row['observed_cost_usd']:.6f} | "
            f"{row['records_with_reported_cost']}/{row['provider_records']} |"
        )
    return "\n".join(lines) + "\n"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _format_optional(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
