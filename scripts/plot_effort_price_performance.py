#!/usr/bin/env python3
"""Create publication-style price/performance plots for an effort sweep."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

EXPECTED_CONFIGURATIONS = 28
EXPECTED_MODELS = 5
EXPECTED_RECORDS_PER_CONFIGURATION = 280
EFFORT_ORDER = ("none", "low", "medium", "high", "xhigh", "max")

MODEL_STYLE = {
    "anthropic/claude-fable-5": ("Claude Fable 5", "#078b75"),
    "anthropic/claude-opus-5": ("Claude Opus 5", "#8a5a44"),
    "google/gemini-3.1-pro-preview": ("Gemini 3.1 Pro", "#c23b70"),
    "google/gemma-4-31b-it": ("Gemma 4 31B", "#777777"),
    "moonshotai/kimi-k3": ("Kimi K3", "#ad7c00"),
    "nvidia/nemotron-3-ultra-550b-a55b": ("Nemotron 3 Ultra", "#00a0b0"),
    "openai/gpt-5.6-sol": ("GPT-5.6-sol", "#dc6b19"),
    "openai/gpt-5.6-terra": ("GPT-5.6-terra", "#3465a4"),
    "openai/gpt-5.6-luna": ("GPT-5.6-luna", "#7656a8"),
    "x-ai/grok-4.5": ("Grok 4.5", "#4f5d2f"),
}

SWEEP_MODELS = {
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-5",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-luna",
}

METRICS = {
    "overall": ("pass_at_1", "Overall Pass@1", 280),
    "core": ("core_pass_at_1", "Core Pass@1", 232),
    "qec": ("qec_pass_at_1", "QEC Pass@1", 48),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Final effort-sweep.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--historical-import-manifest",
        type=Path,
        help="Historical max import manifest; defaults to <campaign>/imported-max/manifest.json",
    )
    parser.add_argument(
        "--supplemental-score-cost",
        type=Path,
        help="Prior completed-run score-cost.json containing additional model families",
    )
    args = parser.parse_args(argv)

    rows = load_configurations(args.source)
    if args.supplemental_score_cost:
        rows.extend(load_supplemental_configurations(args.supplemental_score_cost, excluded_models=SWEEP_MODELS))
    import_manifest = args.historical_import_manifest or args.source.parent.parent / "imported-max" / "manifest.json"
    historical_ids = load_historical_configuration_ids(import_manifest)
    outputs = write_plots(rows, args.out_dir, historical_ids=historical_ids)
    for output in outputs:
        print(output)
    return 0


def load_configurations(  # noqa: C901 - validates every published metric field
    path: Path,
) -> list[dict[str, Any]]:
    """Load and validate the finalized five-model sweep table."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("configurations") if isinstance(payload, Mapping) else None
    acceptance = payload.get("acceptance") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or not isinstance(acceptance, Mapping):
        raise ValueError(f"{path}: expected an effort-sweep report")
    if acceptance.get("publication_ready") is not True:
        raise ValueError(f"{path}: report is not publication-ready")
    if len(rows) != EXPECTED_CONFIGURATIONS:
        raise ValueError(f"{path}: expected {EXPECTED_CONFIGURATIONS} configurations, found {len(rows)}")
    if len({str(row.get("model_id")) for row in rows}) != EXPECTED_MODELS:
        raise ValueError(f"{path}: expected {EXPECTED_MODELS} base models")
    if {str(row.get("model_id")) for row in rows} != SWEEP_MODELS:
        raise ValueError(f"{path}: model roster does not match the finalized sweep")

    seen = set()
    for row in rows:
        configuration_id = str(row.get("configuration_id"))
        if configuration_id in seen:
            raise ValueError(f"{path}: duplicate configuration {configuration_id}")
        seen.add(configuration_id)
        if row.get("records") != EXPECTED_RECORDS_PER_CONFIGURATION:
            raise ValueError(f"{path}: {configuration_id} does not have 280 records")
        if row.get("cost_covered_records") != EXPECTED_RECORDS_PER_CONFIGURATION:
            raise ValueError(f"{path}: {configuration_id} does not have complete cost coverage")
        cost = row.get("provider_cost_per_task_usd")
        if not _finite_positive(cost):
            raise ValueError(f"{path}: {configuration_id} has an invalid average cost")
        for metric, _label, _tasks in METRICS.values():
            score = row.get(metric)
            if not _finite_rate(score):
                raise ValueError(f"{path}: {configuration_id} has an invalid {metric}")
    return rows


def load_supplemental_configurations(  # noqa: C901 - validates every supplemental metric field
    path: Path, *, excluded_models: set[str]
) -> list[dict[str, Any]]:
    """Load additional completed-run models and reproduce their Core/QEC rates."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a score-cost table")
    rows = []
    seen = set()
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}: score-cost rows must be objects")
        model_id = str(item.get("model"))
        if model_id in excluded_models:
            continue
        if model_id in seen:
            raise ValueError(f"{path}: duplicate supplemental model {model_id}")
        if model_id not in MODEL_STYLE:
            raise ValueError(f"{path}: unknown supplemental model {model_id}")
        seen.add(model_id)
        source = _resolve_source_path(path, str(item.get("source")))
        score_rows = _read_score_records(source)
        if len(score_rows) != EXPECTED_RECORDS_PER_CONFIGURATION:
            raise ValueError(f"{source}: expected 280 regraded records")
        suites: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in score_rows:
            suites[str(record.get("suite"))].append(record)
        if {name: len(records) for name, records in suites.items()} != {"core": 232, "qec": 48}:
            raise ValueError(f"{source}: Core/QEC denominators do not match the frozen benchmark")
        overall = sum(_record_passed(record) for record in score_rows) / len(score_rows)
        reported_score = float(item.get("score"))
        if not math.isclose(overall, reported_score, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{source}: reproduced score does not match score-cost.json")
        cost = item.get("average_cost_per_task_usd")
        covered = item.get("records_with_reported_cost")
        if not _finite_positive(cost) or covered != EXPECTED_RECORDS_PER_CONFIGURATION:
            raise ValueError(f"{path}: {model_id} has incomplete or invalid cost data")
        effort = _effort_from_source(source)
        rows.append(
            {
                "configuration_id": f"{model_id.replace('/', '-').replace('.', '-')}__prior-{effort}",
                "model_id": model_id,
                "reasoning_effort": effort,
                "records": len(score_rows),
                "pass_at_1": overall,
                "core_pass_at_1": sum(_record_passed(record) for record in suites["core"]) / len(suites["core"]),
                "qec_pass_at_1": sum(_record_passed(record) for record in suites["qec"]) / len(suites["qec"]),
                "provider_cost_per_task_usd": float(cost),
                "cost_covered_records": int(covered),
                "supplemental_prior_run": True,
            }
        )
    return sorted(rows, key=lambda row: str(row["model_id"]))


def load_historical_configuration_ids(path: Path) -> set[str]:
    """Return configuration IDs imported from the historical max campaign."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts") if isinstance(payload, Mapping) else None
    if payload.get("schema_version") != "qceval.historical_max_import.v1" or not isinstance(artifacts, list):
        raise ValueError(f"{path}: expected a historical max import manifest")
    ids = {str(artifact.get("configuration_id")) for artifact in artifacts if isinstance(artifact, Mapping)}
    if len(ids) != 3 or any(not configuration_id.endswith("__effort-max") for configuration_id in ids):
        raise ValueError(f"{path}: expected the three reused max configurations")
    return ids


def pareto_frontier(rows: Sequence[Mapping[str, Any]], score_name: str) -> list[Mapping[str, Any]]:
    """Return cost-minimizing, score-maximizing non-dominated points."""
    frontier: list[Mapping[str, Any]] = []
    best_score = float("-inf")
    ordered = sorted(
        rows,
        key=lambda row: (float(row["provider_cost_per_task_usd"]), -float(row[score_name])),
    )
    for row in ordered:
        score = float(row[score_name])
        if score > best_score:
            frontier.append(row)
            best_score = score
    return frontier


def write_plots(
    rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    *,
    historical_ids: set[str],
) -> list[Path]:
    """Write overall, Core, and QEC price/performance PNGs."""
    configuration_ids = {str(row["configuration_id"]) for row in rows}
    if not historical_ids <= configuration_ids:
        raise ValueError("historical import manifest contains configurations outside the report")
    unknown_models = {str(row["model_id"]) for row in rows} - set(MODEL_STYLE)
    if unknown_models:
        raise ValueError(f"plot styles are missing for models: {sorted(unknown_models)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for slug, (score_name, score_label, scoring_tasks) in METRICS.items():
        output = out_dir / f"price-vs-performance-{slug}.png"
        _plot(
            rows,
            score_name,
            score_label,
            scoring_tasks,
            output,
            historical_ids=historical_ids,
        )
        outputs.append(output)
    return outputs


def _plot(
    rows: Sequence[Mapping[str, Any]],
    score_name: str,
    score_label: str,
    scoring_tasks: int,
    output: Path,
    *,
    historical_ids: set[str],
) -> None:
    costs = [float(row["provider_cost_per_task_usd"]) for row in rows]
    scores = [float(row[score_name]) for row in rows]
    x_min = 10 ** (math.log10(min(costs)) - 0.16)
    x_max = 10 ** (math.log10(max(costs)) + 0.16)
    y_min = max(0.0, min(scores) - 0.065)
    y_max = min(1.0, max(scores) + 0.045)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.labelweight": "semibold",
        }
    )
    figure, axis = plt.subplots(figsize=(16, 10), dpi=200)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#fbfcfe")

    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model_id"])].append(row)
    for model_id, model_rows in sorted(by_model.items()):
        _display_name, color = MODEL_STYLE[model_id]
        ordered = sorted(model_rows, key=lambda row: _effort_index(str(row["reasoning_effort"])))
        marker = "s" if any(row.get("supplemental_prior_run") is True for row in ordered) else "o"
        axis.plot(
            [float(row["provider_cost_per_task_usd"]) for row in ordered],
            [float(row[score_name]) for row in ordered],
            color=color,
            linewidth=1.45,
            alpha=0.45,
            zorder=2,
        )
        axis.scatter(
            [float(row["provider_cost_per_task_usd"]) for row in ordered],
            [float(row[score_name]) for row in ordered],
            s=64,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=1.1,
            alpha=0.94,
            zorder=4,
        )

    frontier = pareto_frontier(rows, score_name)
    axis.plot(
        [float(row["provider_cost_per_task_usd"]) for row in frontier],
        [float(row[score_name]) for row in frontier],
        color="#087e8b",
        linewidth=1.8,
        linestyle=(0, (4, 3)),
        alpha=0.78,
        zorder=3,
    )
    axis.scatter(
        [float(row["provider_cost_per_task_usd"]) for row in frontier],
        [float(row[score_name]) for row in frontier],
        s=105,
        marker="D",
        facecolor="#087e8b",
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )

    historical_rows = [row for row in rows if str(row["configuration_id"]) in historical_ids]
    axis.scatter(
        [float(row["provider_cost_per_task_usd"]) for row in historical_rows],
        [float(row[score_name]) for row in historical_rows],
        s=175,
        marker="o",
        facecolor="none",
        edgecolor="#111827",
        linewidth=1.3,
        zorder=6,
    )

    axis.set_xscale("log")
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_xticks([0.0002, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
    axis.xaxis.set_major_formatter(FuncFormatter(_format_cost_tick))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.0%}"))
    axis.grid(which="major", color="#dce2e8", linewidth=0.85, alpha=0.9)
    axis.grid(which="minor", axis="x", color="#eef1f4", linewidth=0.55, alpha=0.82)
    axis.set_axisbelow(True)

    _annotate_rows(axis, rows, score_name, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    axis.set_title(
        f"Pass@1 Price vs Performance — {score_label.removesuffix(' Pass@1')}",
        fontsize=22,
        loc="left",
        pad=32,
    )
    axis.text(
        0,
        1.018,
        f"Offline-regraded score versus average provider-reported cost per task across "
        f"{len(by_model)} models; labels show effort, score, and cost",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        color="#56616d",
    )
    axis.set_xlabel("Average provider-reported cost per task, USD (log scale)", labelpad=13)
    axis.set_ylabel(score_label, labelpad=13)

    handles = []
    for model_id, (display_name, color) in MODEL_STYLE.items():
        if model_id not in by_model:
            continue
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="o" if model_id in SWEEP_MODELS else "s",
                markeredgecolor="white",
                linewidth=1.5,
                label=display_name,
            )
        )
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#087e8b",
                marker="D",
                markeredgecolor="white",
                linestyle=(0, (4, 3)),
                label="Price/performance frontier",
            ),
            Line2D(
                [0],
                [0],
                color="#111827",
                marker="o",
                markerfacecolor="none",
                linestyle="none",
                label="Reused historical max",
            ),
        ]
    )
    legend = axis.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#dce2e8",
        framealpha=0.97,
        fontsize=9.2,
        ncol=3,
    )
    legend.get_frame().set_linewidth(0.8)

    figure.text(
        0.072,
        0.026,
        f"{len(rows)} configurations · {len(by_model)} models · {scoring_tasks} scoring tasks/configuration · "
        f"{sum(int(row['cost_covered_records']) for row in rows):,}/"
        f"{sum(int(row['records']) for row in rows):,} provider-cost coverage",
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#68737e",
    )
    figure.text(
        0.072,
        0.009,
        "Squares: completed prior-run model points · Reused max: Fable, Opus, Sol · "
        "Fable and Sol max use historical endpoints",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#7a838c",
    )
    figure.subplots_adjust(left=0.095, right=0.975, top=0.865, bottom=0.13)
    figure.savefig(output, dpi=200, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)


def _annotate_rows(
    axis: Any,
    rows: Sequence[Mapping[str, Any]],
    score_name: str,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    points = [
        (
            (math.log10(float(row["provider_cost_per_task_usd"])) - math.log10(x_min))
            / (math.log10(x_max) - math.log10(x_min)),
            (float(row[score_name]) - y_min) / (y_max - y_min),
        )
        for row in rows
    ]
    occupied: list[tuple[float, float, float, float]] = []
    ordered = sorted(
        zip(rows, points, strict=True),
        key=lambda item: (-item[1][1], item[1][0]),
    )
    for row, point in ordered:
        effort = str(row["reasoning_effort"])
        score = float(row[score_name])
        cost = float(row["provider_cost_per_task_usd"])
        value_label = f"{score:.1%} · ${cost:.4f}"
        model_id = str(row["model_id"])
        first_line = effort
        if row.get("supplemental_prior_run") is True:
            first_line = f"{MODEL_STYLE[model_id][0]} · {effort}"
        label = f"{first_line}\n{value_label}"
        width = min(0.16, 0.0043 * max(len(first_line), len(value_label)))
        height = 0.044
        box = _choose_label_box(point, width, height, occupied, points)
        occupied.append(box)
        label_x, label_y, _right, _top = box
        color = MODEL_STYLE[model_id][1]
        distance = math.hypot(label_x - point[0], label_y - point[1])
        arrowprops = None
        if distance > 0.035:
            arrowprops = {
                "arrowstyle": "-",
                "color": "#97a1ad",
                "linewidth": 0.55,
                "alpha": 0.7,
                "shrinkA": 1,
                "shrinkB": 4,
            }
        axis.annotate(
            label,
            xy=point,
            xycoords=axis.transAxes,
            xytext=(label_x, label_y),
            textcoords=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.6,
            color=color,
            linespacing=1.15,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            arrowprops=arrowprops,
            annotation_clip=False,
            zorder=7,
        )


def _choose_label_box(
    point: tuple[float, float],
    width: float,
    height: float,
    occupied: Sequence[tuple[float, float, float, float]],
    points: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float]:
    px, py = point
    candidates = []
    for vertical_distance in (0.014, 0.055, 0.095, 0.14, 0.19):
        candidates.extend(
            [
                (px + 0.010, py + vertical_distance),
                (px + 0.010, py - height - vertical_distance),
                (px - width - 0.010, py + vertical_distance),
                (px - width - 0.010, py - height - vertical_distance),
                (px - width / 2, py + vertical_distance),
                (px - width / 2, py - height - vertical_distance),
            ]
        )
    best_box = None
    best_penalty = float("inf")
    for left, bottom in candidates:
        left = min(max(left, 0.002), 0.998 - width)
        bottom = min(max(bottom, 0.002), 0.998 - height)
        box = (left, bottom, left + width, bottom + height)
        overlap = sum(_overlap_area(box, other) for other in occupied)
        covered_points = sum(_box_contains(box, other) for other in points if other != point)
        connector = math.hypot(left - px, bottom - py)
        penalty = overlap * 5000 + covered_points * 3 + connector
        if penalty < best_penalty:
            best_box = box
            best_penalty = penalty
    assert best_box is not None
    return best_box


def _overlap_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _box_contains(box: tuple[float, float, float, float], point: tuple[float, float]) -> int:
    padding_x = 0.004
    padding_y = 0.007
    return int(
        box[0] - padding_x <= point[0] <= box[2] + padding_x and box[1] - padding_y <= point[1] <= box[3] + padding_y
    )


def _format_cost_tick(value: float, _position: int) -> str:
    if value <= 0:
        return ""
    return f"${value:g}"


def _resolve_source_path(score_cost_path: Path, source_value: str) -> Path:
    source = Path(source_value)
    if source.is_absolute() and source.is_file():
        return source
    for root in (Path.cwd(), *score_cost_path.resolve().parents):
        candidate = root / source
        if candidate.is_file():
            return candidate
    raise ValueError(f"{score_cost_path}: supplemental source does not exist: {source_value}")


def _read_score_records(path: Path) -> list[Mapping[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number}: expected an object")
        if payload.get("kind") != "summary":
            records.append(payload)
    return records


def _record_passed(record: Mapping[str, Any]) -> bool:
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return False
    verified = evaluation.get("verified_status")
    if verified is None:
        grader = evaluation.get("grader_details")
        verified = grader.get("verified_status") if isinstance(grader, Mapping) else None
    return verified == "verified_pass"


def _effort_from_source(path: Path) -> str:
    parts = path.name.split("__")
    if len(parts) < 3:
        raise ValueError(f"{path}: cannot recover the prior-run effort label")
    return parts[-2]


def _effort_index(effort: str) -> int:
    return EFFORT_ORDER.index(effort) if effort in EFFORT_ORDER else len(EFFORT_ORDER)


def _finite_positive(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed > 0


def _finite_rate(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and 0 <= parsed <= 1


if __name__ == "__main__":
    raise SystemExit(main())
