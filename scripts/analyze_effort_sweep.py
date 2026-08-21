#!/usr/bin/env python3
"""Validate and analyze the complete 28-configuration full-effort sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qceval.production.campaign import configuration_id

EFFORTS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "anthropic/claude-fable-5": ("low", "medium", "high", "xhigh", "max"),
    "anthropic/claude-opus-5": ("low", "medium", "high", "xhigh", "max"),
    "openai/gpt-5.6-sol": ("none", "low", "medium", "high", "xhigh", "max"),
    "openai/gpt-5.6-terra": ("none", "low", "medium", "high", "xhigh", "max"),
    "openai/gpt-5.6-luna": ("none", "low", "medium", "high", "xhigh", "max"),
}
CONFIGURATION_COUNT = sum(len(efforts) for efforts in EFFORTS_BY_MODEL.values())

BOOTSTRAP_SEED = 20260811
BOOTSTRAP_SAMPLES = 10_000
FAILURE_CAUSES = (
    "verifier rejection",
    "malformed or missing candidate",
    "full-ceiling length outcome",
    "structured provider safeguard",
    "explicit textual model refusal",
    "unresolved infrastructure failure",
)
_REFUSAL_PATTERNS = (
    re.compile(r"\bI (?:cannot|can't) (?:assist|help|provide|comply)\b", re.IGNORECASE),
    re.compile(r"\bI am unable to (?:assist|help|provide|comply)\b", re.IGNORECASE),
    re.compile(r"\bI'm unable to (?:assist|help|provide|comply)\b", re.IGNORECASE),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--refusal-audit", type=Path)
    parser.add_argument("--resource-limit-audit", type=Path)
    args = parser.parse_args(argv)
    try:
        report = analyze(
            args.inputs,
            refusal_audit=args.refusal_audit,
            resource_limit_audit=args.resource_limit_audit,
            bootstrap_samples=BOOTSTRAP_SAMPLES,
            require_complete_campaign=True,
        )
    except ValueError as exc:
        parser.error(str(exc))
    write_outputs(report, args.out_dir)
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
    return 0 if report["acceptance"]["publication_ready"] else 2


def analyze(
    inputs: Sequence[Path],
    *,
    refusal_audit: Path | None = None,
    resource_limit_audit: Path | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    require_complete_campaign: bool = True,
) -> dict[str, Any]:
    """Return configuration metrics, paired deltas, and exclusive failures."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    audits = _read_refusal_audit(refusal_audit)
    resource_audits = _read_resource_limit_audit(resource_limit_audit)
    used_resource_audits: set[tuple[str, str, str, str]] = set()
    configurations: list[dict[str, Any]] = []
    task_scores: dict[str, dict[tuple[str, str, str], int]] = {}
    failure_rows: list[dict[str, Any]] = []
    refusal_matches: list[dict[str, Any]] = []
    seen_configs: set[str] = set()
    for path in inputs:
        records, summary = _read_run(path)
        config = _validate_configuration(records, summary, path=path, require_280=require_complete_campaign)
        config_id = config["configuration_id"]
        if config_id in seen_configs:
            raise ValueError(f"duplicate configuration artifact: {config_id}")
        seen_configs.add(config_id)
        metrics, scores, failures, matches, used = _configuration_metrics(config, records, audits, resource_audits)
        configurations.append(metrics)
        task_scores[config_id] = scores
        failure_rows.extend(failures)
        refusal_matches.extend(matches)
        used_resource_audits.update(used)
    if require_complete_campaign:
        expected = {
            configuration_id(model_id, effort) for model_id, efforts in EFFORTS_BY_MODEL.items() for effort in efforts
        }
        if seen_configs != expected:
            missing = sorted(expected - seen_configs)
            extra = sorted(seen_configs - expected)
            raise ValueError(f"campaign configuration set is incomplete; missing={missing[:3]} extra={extra[:3]}")
    pending_matches = [match for match in refusal_matches if match["audit_decision"] == "pending_manual_audit"]
    unused_resource_audits = set(resource_audits) - used_resource_audits
    if unused_resource_audits:
        raise ValueError(f"resource-limit audit contains unmatched keys: {sorted(unused_resource_audits)[:3]}")
    paired = _paired_effort_deltas(task_scores, bootstrap_samples=bootstrap_samples)
    configurations.sort(key=lambda row: (row["model_id"], _effort_index(row["model_id"], row["reasoning_effort"])))
    failure_summary = _failure_summary(failure_rows, configurations)
    return {
        "schema_version": "qceval.full_effort_analysis.v1",
        "bootstrap": {
            "method": "paired_task_cluster_percentile_bootstrap",
            "cluster": "same suite task across four frameworks",
            "samples": bootstrap_samples,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "failure_taxonomy": {
            "mutually_exclusive": True,
            "causes": list(FAILURE_CAUSES),
            "textual_refusal_rule": "frozen_high_precision_v1",
        },
        "configurations": configurations,
        "paired_effort_deltas": paired,
        "failure_causes": failure_summary,
        "textual_refusal_matches": refusal_matches,
        "candidate_resource_limit_adjudications": [
            {
                "configuration_id": key[0],
                "suite": key[1],
                "framework": key[2],
                "task_id": key[3],
                "audit_decision": resource_audits[key],
            }
            for key in sorted(used_resource_audits)
        ],
        "acceptance": {
            "base_models": len({row["model_id"] for row in configurations}),
            "configurations": len(configurations),
            "records": sum(row["records"] for row in configurations),
            "provider_cost_coverage": sum(row["cost_covered_records"] for row in configurations),
            "unresolved_infrastructure_failures": sum(
                row["counts"]["unresolved infrastructure failure"] for row in failure_summary
            ),
            "pending_textual_refusal_audits": len(pending_matches),
            "confirmed_candidate_resource_limits": len(used_resource_audits),
            "publication_ready": (
                len(configurations) == CONFIGURATION_COUNT
                and all(row["records"] == 280 and row["cost_covered_records"] == 280 for row in configurations)
                and not pending_matches
                and not any(row["counts"]["unresolved infrastructure failure"] for row in failure_summary)
            ),
        },
    }


def _validate_configuration(
    records: list[dict[str, Any]],
    summary: Mapping[str, Any],
    *,
    path: Path,
    require_280: bool,
) -> dict[str, str]:
    _require_configuration_size(records, path=path, require_280=require_280)
    identity = _singular_configuration(records, path=path)
    _require_verified_shared_route(records, path=path)
    keys = {(str(record.get("suite")), str(record.get("framework")), str(record.get("task_id"))) for record in records}
    if len(keys) != len(records):
        raise ValueError(f"{path}: duplicate logical task records")
    if require_280:
        _require_full_denominators(records, path=path)
    return identity


def _require_configuration_size(records: list[dict[str, Any]], *, path: Path, require_280: bool) -> None:
    if require_280 and len(records) != 280:
        raise ValueError(f"{path}: configuration artifact must contain exactly 280 records")
    if not records:
        raise ValueError(f"{path}: configuration artifact is empty")


def _singular_configuration(records: list[dict[str, Any]], *, path: Path) -> dict[str, str]:
    model_ids = {str(record.get("model")) for record in records}
    config_ids = {str(_route(record).get("configuration_id")) for record in records}
    reported_efforts = {
        str(value)
        for record in records
        if (value := ((record.get("provider_response") or {}).get("metadata") or {}).get("reasoning_effort"))
        is not None
    }
    if len(model_ids) != 1 or len(config_ids) != 1:
        raise ValueError(f"{path}: model or configuration provenance is not singular")
    model_id = next(iter(model_ids))
    config_id = next(iter(config_ids))
    matching_efforts = [
        effort for effort in EFFORTS_BY_MODEL.get(model_id, ()) if config_id == configuration_id(model_id, effort)
    ]
    if len(matching_efforts) != 1:
        raise ValueError(f"{path}: configuration_id does not identify one model effort")
    effort = matching_efforts[0]
    if reported_efforts - {effort}:
        raise ValueError(f"{path}: reported effort conflicts with configuration identity")
    return {"configuration_id": config_id, "model_id": model_id, "reasoning_effort": effort}


def _require_verified_shared_route(records: list[dict[str, Any]], *, path: Path) -> None:
    routes = [_route(record) for record in records]
    route_signatures = {
        json.dumps(
            {
                name: route.get(name)
                for name in (
                    "endpoint_tag",
                    "max_output_tokens",
                    "output_limit_source",
                    "endpoint_cap_status",
                    "output_token_parameter",
                    "route_revision",
                    "temperature",
                )
            },
            sort_keys=True,
        )
        for route in routes
    }
    if len(route_signatures) != 1 or any(route.get("route_verified") is not True for route in routes):
        raise ValueError(f"{path}: records do not share one verified route identity")


def _require_full_denominators(records: list[dict[str, Any]], *, path: Path) -> None:
    framework_counts = Counter(str(record.get("framework")) for record in records)
    suite_counts = Counter(str(record.get("suite")) for record in records)
    if set(framework_counts.values()) != {70} or suite_counts != {"core": 232, "qec": 48}:
        raise ValueError(f"{path}: framework or Core/QEC denominators are invalid")


def _configuration_metrics(
    config: Mapping[str, str],
    records: list[dict[str, Any]],
    audits: Mapping[tuple[str, str, str, str], str],
    resource_audits: Mapping[tuple[str, str, str, str], str],
) -> tuple[
    dict[str, Any],
    dict[tuple[str, str, str], int],
    list[dict[str, Any]],
    list[dict[str, Any]],
    set[tuple[str, str, str, str]],
]:
    config_id = config["configuration_id"]
    scores: dict[tuple[str, str, str], int] = {}
    failures = []
    refusal_matches = []
    used_resource_audits: set[tuple[str, str, str, str]] = set()
    costs = []
    prompt_tokens = completion_tokens = reasoning_tokens = 0
    latencies = []
    endpoint_tags = Counter()
    for record in records:
        key = (str(record.get("suite")), str(record.get("framework")), str(record.get("task_id")))
        passed = _passed(record)
        scores[key] = int(passed)
        response = record.get("provider_response") or {}
        usage = response.get("usage") or {}
        cost = _finite_nonnegative(usage.get("cost_usd"))
        if cost is not None:
            costs.append(cost)
        prompt_tokens += _nonnegative_int(usage.get("prompt_tokens"))
        completion_tokens += _nonnegative_int(usage.get("completion_tokens"))
        reasoning_tokens += _nonnegative_int(usage.get("reasoning_tokens"))
        latency = _attempt_latency_seconds(response)
        if latency is not None:
            latencies.append(latency)
        endpoint_tags[str(_route(record).get("endpoint_tag"))] += 1
        match = _textual_refusal_match(record)
        audit_key = (config_id, key[0], key[1], key[2])
        decision = audits.get(audit_key, "pending_manual_audit") if match else "not_applicable"
        if match:
            refusal_matches.append(
                {
                    "configuration_id": config_id,
                    "suite": key[0],
                    "framework": key[1],
                    "task_id": key[2],
                    "rule_match": match,
                    "audit_decision": decision,
                }
            )
        resource_decision = resource_audits.get(audit_key)
        if resource_decision is not None:
            if not _candidate_resource_limit_timeout(record):
                raise ValueError(f"resource-limit audit key does not identify an evaluator timeout: {audit_key}")
            used_resource_audits.add(audit_key)
        cause = _failure_cause(
            record,
            passed=passed,
            refusal_confirmed=decision == "confirmed_refusal",
            candidate_resource_limit=resource_decision == "confirmed_candidate_resource_limit",
        )
        if cause is not None:
            failures.append(
                {
                    "configuration_id": config_id,
                    "model_id": config["model_id"],
                    "reasoning_effort": config["reasoning_effort"],
                    "suite": key[0],
                    "framework": key[1],
                    "task_id": key[2],
                    "cause": cause,
                }
            )
    route = _route(records[0])
    return (
        {
            **dict(config),
            "records": len(records),
            "pass_at_1": sum(scores.values()) / len(records),
            "core_pass_at_1": _rate(records, lambda record: record.get("suite") == "core"),
            "qec_pass_at_1": _rate(records, lambda record: record.get("suite") == "qec"),
            "framework_pass_at_1": {
                framework: _rate(records, lambda record, framework=framework: record.get("framework") == framework)
                for framework in ("qiskit", "cirq", "pennylane", "cudaq")
            },
            "provider_cost_usd": sum(costs),
            "provider_cost_per_task_usd": sum(costs) / len(records) if len(costs) == len(records) else None,
            "cost_covered_records": len(costs),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "average_completion_tokens": completion_tokens / len(records),
            "average_reasoning_tokens": reasoning_tokens / len(records),
            "average_latency_seconds": sum(latencies) / len(latencies) if len(latencies) == len(records) else None,
            "latency_covered_records": len(latencies),
            "endpoint_usage": dict(sorted(endpoint_tags.items())),
            "route_revision": route.get("route_revision"),
            "output_token_parameter": route.get("output_token_parameter"),
            "configured_output_tokens": route.get("max_output_tokens"),
            "temperature_behavior": "explicit_zero" if route.get("temperature") == 0.0 else "not_exposed",
        },
        scores,
        failures,
        refusal_matches,
        used_resource_audits,
    )


def _failure_cause(
    record: Mapping[str, Any],
    *,
    passed: bool,
    refusal_confirmed: bool,
    candidate_resource_limit: bool,
) -> str | None:
    if passed:
        return None
    if candidate_resource_limit:
        return "verifier rejection"
    if record.get("status") == "infrastructure_error":
        return "unresolved infrastructure failure"
    if _structured_safeguard(record):
        return "structured provider safeguard"
    if refusal_confirmed:
        return "explicit textual model refusal"
    metadata = (record.get("provider_response") or {}).get("metadata") or {}
    route = metadata.get("route") or {}
    if metadata.get("finish_reason") == "length" and _completion_tokens(record) == route.get("max_output_tokens"):
        return "full-ceiling length outcome"
    response = record.get("provider_response") or {}
    if not response.get("code") or record.get("status") in {"provider_failed", "compile_failed"}:
        return "malformed or missing candidate"
    return "verifier rejection"


def _candidate_resource_limit_timeout(record: Mapping[str, Any]) -> bool:
    evaluation = record.get("evaluation") or {}
    error = evaluation.get("error") if isinstance(evaluation, Mapping) else None
    return (
        record.get("status") in {"failed", "infrastructure_error"}
        and isinstance(evaluation, Mapping)
        and evaluation.get("verified_status") == "resource_limit"
        and evaluation.get("error_type") in {"EvaluationTimeout", "InfrastructureError"}
        and isinstance(error, str)
        and error.startswith("evaluation timed out after ")
    )


def _structured_safeguard(record: Mapping[str, Any]) -> bool:
    response = record.get("provider_response") or {}
    raw = response.get("raw_response") or {}
    choices = raw.get("choices") if isinstance(raw, Mapping) else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else {}
    finish = choice.get("finish_reason") or choice.get("native_finish_reason")
    if finish in {"content_filter", "safety", "blocked"}:
        return True
    error = raw.get("error") if isinstance(raw, Mapping) else None
    metadata = error.get("metadata") if isinstance(error, Mapping) else None
    return isinstance(metadata, Mapping) and (
        metadata.get("is_safety") is True or metadata.get("reason") in {"content_policy", "safety", "blocked"}
    )


def _textual_refusal_match(record: Mapping[str, Any]) -> str | None:
    raw = (record.get("provider_response") or {}).get("raw_response") or {}
    choices = raw.get("choices") if isinstance(raw, Mapping) else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else None
    message = choice.get("message") if isinstance(choice, Mapping) else None
    text = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(text, str):
        return None
    for pattern in _REFUSAL_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _paired_effort_deltas(
    task_scores: Mapping[str, Mapping[tuple[str, str, str], int]],
    *,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    rows = []
    for model_id, efforts in EFFORTS_BY_MODEL.items():
        for first_index, first in enumerate(efforts):
            for second in efforts[first_index + 1 :]:
                first_scores = task_scores.get(configuration_id(model_id, first))
                second_scores = task_scores.get(configuration_id(model_id, second))
                if first_scores is None or second_scores is None:
                    continue
                if set(first_scores) != set(second_scores):
                    raise ValueError(f"{model_id}: effort pair does not contain identical logical tasks")
                deltas = {key: second_scores[key] - first_scores[key] for key in first_scores}
                observed = sum(deltas.values()) / len(deltas)
                low, high = _cluster_bootstrap_interval(deltas, samples=bootstrap_samples)
                rows.append(
                    {
                        "model_id": model_id,
                        "effort_a": first,
                        "effort_b": second,
                        "delta_b_minus_a": observed,
                        "ci_95_low": low,
                        "ci_95_high": high,
                        "logical_tasks": len(deltas),
                        "clusters": len({(suite, task_id) for suite, _, task_id in deltas}),
                    }
                )
    return rows


def _cluster_bootstrap_interval(
    deltas: Mapping[tuple[str, str, str], int],
    *,
    samples: int,
) -> tuple[float, float]:
    by_cluster: dict[tuple[str, str], list[int]] = defaultdict(list)
    for (suite, _framework, task_id), delta in deltas.items():
        by_cluster[(suite, task_id)].append(delta)
    clusters = sorted(by_cluster)
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(samples):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        values = [value for cluster in sampled for value in by_cluster[cluster]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    return (_quantile(estimates, 0.025), _quantile(estimates, 0.975))


def _failure_summary(
    failure_rows: Sequence[Mapping[str, Any]], configurations: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = {row["configuration_id"]: Counter() for row in configurations}
    for row in failure_rows:
        counts[str(row["configuration_id"])][str(row["cause"])] += 1
    return [
        {
            "configuration_id": row["configuration_id"],
            "model_id": row["model_id"],
            "reasoning_effort": row["reasoning_effort"],
            "counts": {cause: counts[row["configuration_id"]][cause] for cause in FAILURE_CAUSES},
        }
        for row in configurations
    ]


def write_outputs(report: Mapping[str, Any], out_dir: Path) -> None:
    """Write JSON, TSV, Markdown, and the five required Matplotlib plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "effort-sweep.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_configuration_tsv(report["configurations"], out_dir / "configurations.tsv")
    _write_dict_tsv(report["paired_effort_deltas"], out_dir / "paired-effort-deltas.tsv")
    _write_failure_tsv(report["failure_causes"], out_dir / "failure-causes.tsv")
    _write_dict_tsv(report["textual_refusal_matches"], out_dir / "textual-refusal-audit.tsv")
    (out_dir / "effort-sweep.md").write_text(_markdown(report), encoding="utf-8")
    _plots(report, out_dir)


def _plots(report: Mapping[str, Any], out_dir: Path) -> None:
    configurations = report["configurations"]
    _line_plot(configurations, "pass_at_1", "Pass@1", out_dir / "effort-versus-pass1.png")
    _scatter_plot(
        configurations,
        "provider_cost_per_task_usd",
        "Average provider cost (USD/task)",
        out_dir / "score-versus-cost.png",
    )
    token_rows = [
        {**row, "reasoning_completion_tokens": row["average_reasoning_tokens"] + row["average_completion_tokens"]}
        for row in configurations
    ]
    _scatter_plot(
        token_rows,
        "reasoning_completion_tokens",
        "Average reasoning + completion tokens",
        out_dir / "score-versus-tokens.png",
    )
    _stacked_failure_plot(report["failure_causes"], out_dir / "failure-causes.png")
    _safeguard_refusal_plot(report["failure_causes"], out_dir / "safeguards-and-refusals.png")


def _line_plot(rows: Sequence[Mapping[str, Any]], y_name: str, y_label: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model_id"])].append(row)
    for model_id, model_rows in sorted(by_model.items()):
        model_rows = sorted(model_rows, key=lambda row: _effort_index(model_id, str(row["reasoning_effort"])))
        axis.plot(
            [str(row["reasoning_effort"]) for row in model_rows],
            [float(row[y_name]) for row in model_rows],
            marker="o",
            label=model_id,
        )
    axis.set_ylabel(y_label)
    axis.set_xlabel("Reasoning effort")
    axis.set_ylim(0, 1)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _scatter_plot(rows: Sequence[Mapping[str, Any]], x_name: str, x_label: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    for row in rows:
        x_value = row.get(x_name)
        if x_value is None:
            continue
        axis.scatter(float(x_value), float(row["pass_at_1"]), label=str(row["configuration_id"]), s=28)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Pass@1")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _stacked_failure_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 7))
    labels = [str(row["configuration_id"]) for row in rows]
    bottom = [0] * len(rows)
    for cause in FAILURE_CAUSES:
        values = [int(row["counts"][cause]) for row in rows]
        axis.bar(labels, values, bottom=bottom, label=cause)
        bottom = [left + right for left, right in zip(bottom, values, strict=True)]
    axis.tick_params(axis="x", rotation=90, labelsize=6)
    axis.set_ylabel("Failed tasks")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _safeguard_refusal_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 6))
    labels = [str(row["configuration_id"]) for row in rows]
    positions = list(range(len(rows)))
    safeguards = [int(row["counts"]["structured provider safeguard"]) for row in rows]
    refusals = [int(row["counts"]["explicit textual model refusal"]) for row in rows]
    axis.bar([position - 0.2 for position in positions], safeguards, 0.4, label="Confirmed safeguards")
    axis.bar([position + 0.2 for position in positions], refusals, 0.4, label="Explicit refusals")
    axis.set_xticks(positions, labels, rotation=90, fontsize=6)
    axis.set_ylabel("Confirmed records")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_configuration_tsv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    flattened = []
    for row in rows:
        flattened.append(
            {
                **{key: value for key, value in row.items() if key not in {"framework_pass_at_1", "endpoint_usage"}},
                **{f"{name}_pass_at_1": value for name, value in row["framework_pass_at_1"].items()},
                "endpoint_usage": json.dumps(row["endpoint_usage"], sort_keys=True),
            }
        )
    _write_dict_tsv(flattened, path)


def _write_failure_tsv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    flattened = [
        {
            "configuration_id": row["configuration_id"],
            "model_id": row["model_id"],
            "reasoning_effort": row["reasoning_effort"],
            **row["counts"],
        }
        for row in rows
    ]
    _write_dict_tsv(flattened, path)


def _write_dict_tsv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, dialect="excel-tab", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Full-effort Pass@1 sweep",
        "",
        "| Configuration | Overall | Core | QEC | USD/task | Reasoning tokens | Completion tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["configurations"]:
        lines.append(
            f"| {row['configuration_id']} | {row['pass_at_1']:.4f} | {row['core_pass_at_1']:.4f} | "
            f"{row['qec_pass_at_1']:.4f} | {_format_optional(row['provider_cost_per_task_usd'], 8)} | "
            f"{row['average_reasoning_tokens']:.1f} | {row['average_completion_tokens']:.1f} |"
        )
    acceptance = report["acceptance"]
    lines.extend(
        [
            "",
            f"Publication ready: **{str(acceptance['publication_ready']).lower()}**.",
            "",
            f"Pending refusal audits: {acceptance['pending_textual_refusal_audits']}; "
            f"unresolved infrastructure failures: {acceptance['unresolved_infrastructure_failures']}.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_run(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    summary = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: record must be an object")
        if payload.get("kind") == "result":
            records.append(payload)
        elif payload.get("kind") == "summary":
            if summary is not None:
                raise ValueError(f"{path}: multiple summaries")
            summary = payload
    if summary is None:
        raise ValueError(f"{path}: missing summary")
    return records, summary


def _read_refusal_audit(path: Path | None) -> dict[tuple[str, str, str, str], str]:
    if path is None:
        return {}
    allowed = {"confirmed_refusal", "rejected_match"}
    rows = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, dialect="excel-tab"):
            decision = str(row.get("audit_decision"))
            if decision not in allowed:
                raise ValueError(f"refusal audit contains invalid decision {decision!r}")
            key = (
                str(row.get("configuration_id")),
                str(row.get("suite")),
                str(row.get("framework")),
                str(row.get("task_id")),
            )
            if key in rows:
                raise ValueError(f"duplicate refusal audit key: {key}")
            rows[key] = decision
    return rows


def _read_resource_limit_audit(path: Path | None) -> dict[tuple[str, str, str, str], str]:
    if path is None:
        return {}
    rows = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, dialect="excel-tab"):
            decision = str(row.get("audit_decision"))
            if decision != "confirmed_candidate_resource_limit":
                raise ValueError(f"resource-limit audit contains invalid decision {decision!r}")
            key = (
                str(row.get("configuration_id")),
                str(row.get("suite")),
                str(row.get("framework")),
                str(row.get("task_id")),
            )
            if key in rows:
                raise ValueError(f"duplicate resource-limit audit key: {key}")
            rows[key] = decision
    return rows


def _route(record: Mapping[str, Any]) -> Mapping[str, Any]:
    response = record.get("provider_response")
    metadata = response.get("metadata") if isinstance(response, Mapping) else None
    route = metadata.get("route") if isinstance(metadata, Mapping) else None
    return route if isinstance(route, Mapping) else {}


def _passed(record: Mapping[str, Any]) -> bool:
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return False
    verified = evaluation.get("verified_status")
    if verified is None:
        grader = evaluation.get("grader_details")
        verified = grader.get("verified_status") if isinstance(grader, Mapping) else None
    return verified == "verified_pass"


def _rate(records: Sequence[Mapping[str, Any]], predicate: Any) -> float:
    selected = [record for record in records if predicate(record)]
    return sum(_passed(record) for record in selected) / len(selected) if selected else 0.0


def _completion_tokens(record: Mapping[str, Any]) -> int | None:
    usage = (record.get("provider_response") or {}).get("usage") or {}
    value = usage.get("completion_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _attempt_latency_seconds(response: Mapping[str, Any]) -> float | None:
    metadata = response.get("metadata") or {}
    history = metadata.get("attempt_history") if isinstance(metadata, Mapping) else None
    accepted = [
        item for item in history or [] if isinstance(item, Mapping) and item.get("status") == "accepted_model_outcome"
    ]
    if len(accepted) != 1:
        return None
    try:
        start = datetime.fromisoformat(str(accepted[0]["started_at_utc"]).replace("Z", "+00:00"))
        finish = datetime.fromisoformat(str(accepted[0]["finished_at_utc"]).replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return None
    elapsed = (finish - start).total_seconds()
    return elapsed if elapsed >= 0 and math.isfinite(elapsed) else None


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _effort_index(model_id: str, effort: str) -> int:
    efforts = EFFORTS_BY_MODEL.get(model_id, ())
    return efforts.index(effort) if effort in efforts else len(efforts)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sample")
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _format_optional(value: Any, digits: int) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
