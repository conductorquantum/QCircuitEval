"""Run-protocol and repeated-attempt report metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from qceval.models import BenchmarkRecord, RunConfig
from qceval.providers.registry import DEFAULT_TEMPERATURE
from qceval.reporting._records import record_verified_status


def pass_at_k(n: int, c: int, k: int) -> float:
    """Return unbiased Pass@K estimate for ``n`` samples and ``c`` correct.

    Args:
        n: Number of generated samples.
        c: Number of passing samples.
        k: Pass@K cutoff.

    Returns:
        Estimated probability that at least one of ``k`` samples passes.

    Raises:
        ValueError: If the counts or cutoff are outside their valid ranges.
    """
    if not 0 <= c <= n:
        raise ValueError(f"c={c} not in [0, n={n}]")
    if not 1 <= k <= n:
        raise ValueError(f"k={k} not in [1, n={n}]")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    product = 1.0
    for index in range(k):
        product *= (n - c - index) / (n - index)
    return 1.0 - product


def run_protocol(run_config: RunConfig | None) -> dict[str, Any]:
    """Describe the sampling and feedback protocol for a run.

    Args:
        run_config: Run configuration, or ``None`` for single-sample defaults.

    Returns:
        Sampling, Pass@K, attempt, and feedback settings.
    """
    max_attempts = 1 if run_config is None else run_config.max_attempts
    protocol = {
        "samples_per_task": 1 if run_config is None else run_config.samples_per_task,
        "pass_k": 1 if run_config is None else run_config.pass_k,
        "max_attempts": max_attempts,
        "max_repairs": max_attempts - 1,
        "attempt_semantics": "one_initial_generation_plus_bounded_repairs",
        "feedback_enabled": max_attempts > 1,
    }
    if run_config is None:
        protocol.update(
            {
                "feedback_max_chars": None,
                "feedback_policy": None,
                "generation_parameters": {
                    "temperature": {"value": None, "source": "unknown"},
                    "reasoning_effort": {"value": None, "source": "unknown"},
                    "reasoning_enabled": {"value": None, "source": "unknown"},
                    "endpoint_tag": {"value": None, "source": "unknown"},
                    "max_output_tokens": {"value": None, "source": "unknown"},
                    "output_limit_source": {"value": None, "source": "unknown"},
                    "endpoint_cap_status": {"value": None, "source": "unknown"},
                    "output_token_parameter": {"value": None, "source": "unknown"},
                    "route_revision": {"value": None, "source": "unknown"},
                    "configuration_id": {"value": None, "source": "unknown"},
                    "seed": {"value": None, "source": "not_supported"},
                },
            }
        )
        return protocol
    protocol.update(
        {
            "feedback_max_chars": run_config.feedback_max_chars,
            "feedback_policy": run_config.feedback_policy.to_dict(),
            "generation_parameters": _generation_parameters(run_config),
        }
    )
    return protocol


def _generation_parameters(run_config: RunConfig) -> dict[str, Any]:
    if run_config.provider == "openrouter":
        configured = run_config.provider_config.get("temperature")
        pinned_endpoint = run_config.provider_config.get("openrouter_endpoint_tag")
        reasoning_effort = run_config.provider_config.get("reasoning_effort")
        reasoning_enabled = run_config.provider_config.get("reasoning_enabled")
        return {
            "temperature": {
                "value": (
                    None
                    if configured is None and pinned_endpoint is not None
                    else DEFAULT_TEMPERATURE
                    if configured is None
                    else configured
                ),
                "source": (
                    "not_exposed"
                    if configured is None and pinned_endpoint is not None
                    else "provider_default"
                    if configured is None
                    else "explicit"
                ),
            },
            "reasoning_effort": {
                "value": reasoning_effort,
                "source": (
                    "explicit"
                    if reasoning_effort is not None
                    else "not_applicable"
                    if reasoning_enabled is not None
                    else "model_default"
                ),
            },
            "reasoning_enabled": {
                "value": reasoning_enabled,
                "source": (
                    "explicit"
                    if reasoning_enabled is not None
                    else "not_applicable"
                    if reasoning_effort is not None
                    else "model_default"
                ),
            },
            "endpoint_tag": {
                "value": pinned_endpoint,
                "source": "explicit" if pinned_endpoint is not None else "unrestricted",
            },
            "max_output_tokens": {
                "value": run_config.provider_config.get("openrouter_max_output_tokens"),
                "source": "explicit" if pinned_endpoint is not None else "not_configured",
            },
            "output_limit_source": {
                "value": run_config.provider_config.get("openrouter_output_limit_source"),
                "source": "explicit" if pinned_endpoint is not None else "not_configured",
            },
            "endpoint_cap_status": {
                "value": run_config.provider_config.get("openrouter_endpoint_cap_status"),
                "source": "explicit" if pinned_endpoint is not None else "not_configured",
            },
            "output_token_parameter": {
                "value": run_config.provider_config.get("openrouter_output_token_parameter"),
                "source": "explicit" if pinned_endpoint is not None else "not_configured",
            },
            "route_revision": {
                "value": run_config.provider_config.get("openrouter_route_revision"),
                "source": "explicit" if pinned_endpoint is not None else "not_configured",
            },
            "configuration_id": {
                "value": run_config.provider_config.get("configuration_id"),
                "source": "explicit" if pinned_endpoint is not None else "not_configured",
            },
            "seed": {"value": None, "source": "not_supported"},
        }
    return {
        "temperature": {"value": None, "source": "not_exposed"},
        "reasoning_effort": {"value": None, "source": "not_exposed"},
        "reasoning_enabled": {"value": None, "source": "not_exposed"},
        "endpoint_tag": {"value": None, "source": "not_exposed"},
        "max_output_tokens": {"value": None, "source": "not_exposed"},
        "output_limit_source": {"value": None, "source": "not_exposed"},
        "endpoint_cap_status": {"value": None, "source": "not_exposed"},
        "output_token_parameter": {"value": None, "source": "not_exposed"},
        "route_revision": {"value": None, "source": "not_exposed"},
        "configuration_id": {"value": None, "source": "not_exposed"},
        "seed": {"value": None, "source": "not_supported"},
    }


def task_totals(records: list[BenchmarkRecord]) -> dict[str, int]:
    """Count unique tasks and physical result records.

    Args:
        records: Benchmark records to aggregate.

    Returns:
        Unique logical-task and physical-record counts.
    """
    unique_tasks = {(record.suite, record.framework, record.task_id) for record in records}
    return {"unique_tasks": len(unique_tasks), "record_count": len(records)}


def pass_at_k_summary(records: list[BenchmarkRecord], k: int) -> dict[str, Any]:
    """Aggregate Pass@K estimates by task.

    Args:
        records: Benchmark records grouped into repeated task samples.
        k: Pass@K cutoff applied to every task group.

    Returns:
        Per-task estimates and aggregate compile and pass metrics.

    Raises:
        ValueError: If ``k`` is invalid for any task's sample count.
    """
    grouped: dict[tuple[str, str, str], list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.suite, record.framework, record.task_id)].append(record)

    task_rows = []
    estimates = []
    compiled_tasks = 0
    rerun_required = 0
    for (suite, framework, task_id), items in sorted(grouped.items()):
        scoreable_items = [record for record in items if record.status != "infrastructure_error"]
        infrastructure_samples = len(items) - len(scoreable_items)
        sample_count = len(scoreable_items)
        correct_count = sum(record_verified_status(record) == "verified_pass" for record in scoreable_items)
        estimate = (
            pass_at_k(sample_count, correct_count, k) if infrastructure_samples == 0 and sample_count >= k else None
        )
        if estimate is not None:
            estimates.append(estimate)
        else:
            rerun_required += 1
        if estimate is not None and any(
            record.evaluation is not None and record.evaluation.compiled for record in scoreable_items
        ):
            compiled_tasks += 1
        task_rows.append(
            {
                "suite": suite,
                "framework": framework,
                "task_id": task_id,
                "assigned_n": len(items),
                "n": sample_count,
                "c": correct_count,
                "estimate": estimate,
                "infrastructure_samples": infrastructure_samples,
                "rerun_required": estimate is None,
            }
        )

    task_count = len(estimates)
    pass_mean = sum(estimates) / task_count if task_count else 0.0
    return {
        "k": k,
        "tasks_evaluated": task_count,
        "tasks_assigned": len(task_rows),
        "tasks_requiring_rerun": rerun_required,
        "compiled_tasks": compiled_tasks,
        "compiled_at_k": compiled_tasks / task_count if task_count else 0.0,
        "expected_passed": sum(estimates),
        "pass_at_k": pass_mean,
        "tasks": task_rows,
    }


def feedback_summary(records: list[BenchmarkRecord], max_attempts: int) -> dict[str, Any]:
    """Aggregate cumulative compile and pass rates across feedback attempts.

    Args:
        records: Benchmark records from initial and feedback attempts.
        max_attempts: Configured maximum number of attempts per sample.

    Returns:
        Attempt-level cumulative rates and final pass rate.
    """
    grouped: dict[tuple[str, str, str, int], list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.suite, record.framework, record.task_id, record.sample_index)].append(record)

    assigned_chains = [sorted(items, key=lambda record: record.attempt_index) for items in grouped.values()]
    infrastructure_chains = [
        chain for chain in assigned_chains if any(record.status == "infrastructure_error" for record in chain)
    ]
    chains = [chain for chain in assigned_chains if chain not in infrastructure_chains]
    total_chains = len(chains)
    max_observed = max((record.attempt_index for record in records), default=0)
    last_level = max(max_observed, max_attempts - 1)
    levels = []
    for attempt_index in range(last_level + 1):
        attempted = sum(any(record.attempt_index == attempt_index for record in chain) for chain in chains)
        compiled_attempts = [_first_compiled(chain) for chain in chains]
        passed_attempts = [_first_passed(chain) for chain in chains]
        first_compiled = sum(value is not None and value <= attempt_index for value in compiled_attempts)
        first_passed = sum(value is not None and value <= attempt_index for value in passed_attempts)
        levels.append(
            {
                "attempt_index": attempt_index,
                "label": "initial" if attempt_index == 0 else f"repair_{attempt_index}",
                "attempted": attempted,
                "first_compiled": first_compiled,
                "first_passed": first_passed,
                "cumulative_compile_rate": first_compiled / total_chains if total_chains else 0.0,
                "cumulative_pass_rate": first_passed / total_chains if total_chains else 0.0,
            }
        )

    final_passed = sum(_first_passed(chain) is not None for chain in chains)
    return {
        "max_attempts": max_attempts,
        "assigned_chains": len(assigned_chains),
        "scoreable_chains": total_chains,
        "infrastructure_chains": len(infrastructure_chains),
        "rerun_required": len(infrastructure_chains),
        "levels": levels,
        "final_pass_rate": final_passed / total_chains if total_chains else 0.0,
    }


def _first_compiled(records: list[BenchmarkRecord]) -> int | None:
    for record in records:
        if record.evaluation is not None and record.evaluation.compiled:
            return record.attempt_index
    return None


def _first_passed(records: list[BenchmarkRecord]) -> int | None:
    for record in records:
        if record_verified_status(record) == "verified_pass":
            return record.attempt_index
    return None
