from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_run_costs import _score, render_markdown, render_tsv, summarize_costs


def test_summarize_costs_scales_reported_openrouter_cost(tmp_path: Path) -> None:
    path = tmp_path / "pass1.jsonl"
    result = {
        "kind": "result",
        "provider_response": {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "reasoning_tokens": 5,
                "cost_usd": 0.25,
            },
            "raw_response": {"usage": {"cost": 9.0}},
        },
    }
    summary = {
        "kind": "summary",
        "model": "model",
        "summary": {
            "pass_rate": 1.0,
            "run_protocol": {"samples_per_task": 1, "pass_k": 1, "max_attempts": 1},
            "task_totals": {"unique_tasks": 1, "record_count": 1},
        },
    }
    path.write_text("\n".join([json.dumps(result), json.dumps(summary)]) + "\n", encoding="utf-8")

    rows = summarize_costs([path], lite_prompts=2, target_prompts=58)

    assert rows[0]["observed_cost_usd"] == 0.25
    assert rows[0]["average_cost_per_task_usd"] == 0.25
    assert rows[0]["estimated_target_cost_usd"] == 7.25
    assert rows[0]["reasoning_tokens"] == 5
    assert rows[0]["score_name"] == "pass_rate"
    assert rows[0]["score"] == 1.0


def test_summarize_costs_supports_legacy_raw_cost_and_marks_missing_coverage(tmp_path: Path) -> None:
    path = tmp_path / "pass1.jsonl"
    records = [
        {
            "kind": "result",
            "provider_response": {"usage": {}, "raw_response": {"usage": {"cost": 0.1}}},
        },
        {"kind": "result", "provider_response": {"usage": {}, "raw_response": None}},
    ]
    summary = {
        "kind": "summary",
        "model": "model",
        "summary": {
            "pass_rate": 0.5,
            "run_protocol": {"samples_per_task": 1, "pass_k": 1, "max_attempts": 1},
            "task_totals": {"unique_tasks": 2, "record_count": 2},
        },
    }
    path.write_text(
        "\n".join(json.dumps(payload) for payload in [*records, summary]) + "\n",
        encoding="utf-8",
    )

    row = summarize_costs([path], lite_prompts=2, target_prompts=58)[0]

    assert row["observed_cost_usd"] == 0.1
    assert row["records_with_reported_cost"] == 1
    assert row["reported_cost_coverage"] == 0.5
    assert row["average_cost_per_task_usd"] is None
    assert row["estimated_target_cost_usd"] is None


def test_score_selects_protocol_specific_endpoint() -> None:
    assert _score({"pass_at_k": {"pass_at_k": 0.75}}, "pass5") == ("pass_at_5", 0.75)
    assert _score(
        {"feedback_lineage": {"terminal_pass_rate": 0.6}},
        "feedback5",
    ) == ("feedback_terminal_pass_rate", 0.6)


def test_score_cost_table_renders_tsv_and_markdown() -> None:
    row = {
        "model": "author/model",
        "protocol": "pass1",
        "score_name": "pass_rate",
        "score": 0.75,
        "average_cost_per_task_usd": 0.0125,
        "observed_cost_usd": 3.5,
        "estimated_target_cost_usd": 3.5,
        "records_with_reported_cost": 280,
        "provider_records": 280,
    }

    assert "author/model\tpass1\tpass_rate\t0.750000\t0.01250000" in render_tsv([row])
    assert "| author/model | pass1 | 0.750000 | 0.01250000 | 3.500000 | 280/280 |" in render_markdown([row])
