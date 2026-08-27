"""Command-line interface for running QCircuitEval benchmarks."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from qceval.cli_plan import (
    ALL_REASONING_TOKEN,
    ReasoningJob,
    apply_reasoning_job,
    expand_reasoning_jobs,
    is_multi_model,
    sweep_manifest_path,
    sweep_output_path,
    write_sweep_manifest,
)
from qceval.cli_types import non_negative_float, positive_float, positive_int
from qceval.core.bench import DEFAULT_FRAMEWORKS, SUPPORTED_FRAMEWORKS, Adaptor
from qceval.core.io import infer_format, write_output
from qceval.core.runner import BenchmarkRunner
from qceval.models import Framework, RunConfig, RunOptions, Suite
from qceval.providers.registry import build_provider, provider_names
from qceval.reports import format_run_summary
from qceval.semantics.contracts import ContractRegistry, contract_hash


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``qceval`` command.

    Args:
        argv: Optional argument vector excluding the program name.  When
            omitted, :mod:`argparse` reads process arguments.

    Returns:
        Process exit code.  ``0`` means success; ``2`` means no supported
        command was selected.
    """
    args = parse_args(argv)
    if args.command == "run":
        return _run_command(args)
    if args.command == "contracts":
        return _contracts(args)
    return 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument vector excluding the program name.

    Returns:
        Parsed argparse namespace.

    Raises:
        SystemExit: If arguments are invalid, as normal for :mod:`argparse`.
    """
    parser, run = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        args.matrix_requested = bool(args.registry or args.reasoning_effort == ALL_REASONING_TOKEN)
        try:
            args.reasoning_jobs = expand_reasoning_jobs(args)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            run.error(str(exc))
        _validate_run_args(args, run)
    return args


def _build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(prog="qceval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run provider against qceval")
    _add_run_arguments(run)
    contracts = subparsers.add_parser("contracts", help="Inspect semantic behavior contracts")
    _add_contract_arguments(contracts)
    return parser, run


def _add_run_arguments(run: argparse.ArgumentParser) -> None:
    run.add_argument("--provider", choices=provider_names(), default="smoke")
    run.add_argument(
        "--framework",
        choices=[*SUPPORTED_FRAMEWORKS, "all"],
        nargs="+",
        action="extend",
        help="One or more frameworks; repeat the flag to add another group.",
    )
    run.add_argument("--suite", choices=["core", "qec", "all"], default="core")
    run.add_argument("--source-hint", help="Optional bundled-source hint")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--output-format", choices=["auto", "json", "jsonl"], default="auto")
    run.add_argument("--model")
    run.add_argument(
        "--registry",
        nargs="+",
        type=Path,
        help="Model registry JSON files; expands each model's listed reasoning efforts.",
    )
    run.add_argument("--max-tasks", type=positive_int)
    run.add_argument("--tasks", nargs="+", type=positive_int, help="Suite-local task numbers to include.")
    run.add_argument(
        "--rerun",
        "--prompt",
        dest="prompt",
        choices=[*SUPPORTED_FRAMEWORKS, "all"],
        nargs="+",
        action="extend",
        help="Frameworks to prompt again. Combine with --regrade to run both phases.",
    )
    run.add_argument(
        "--regrade",
        choices=[*SUPPORTED_FRAMEWORKS, "all"],
        nargs="+",
        action="extend",
        help="Frameworks to grade. Regrade-only frameworks read stored code from --input.",
    )
    run.add_argument(
        "--input",
        type=Path,
        help="JSON or JSONL run artifact supplying code for regrade-only frameworks.",
    )
    run.add_argument("--smoke-mode", choices=["canonical", "empty", "error"])
    openrouter_credentials = run.add_mutually_exclusive_group()
    openrouter_credentials.add_argument(
        "--openrouter-api-key",
        help="OpenRouter credential; falls back to OPENROUTER_API_KEY and then .env.",
    )
    openrouter_credentials.add_argument(
        "--openrouter-api-key-file",
        type=Path,
        help="Read the OpenRouter API key from a file instead of exposing it in process arguments.",
    )
    run.add_argument("--openrouter-base-url")
    run.add_argument(
        "--openrouter-endpoint-tag",
        help="Exact OpenRouter endpoint tag. Pins provider.only to this singular route and disables fallbacks.",
    )
    run.add_argument(
        "--openrouter-max-output-tokens",
        type=positive_int,
        help="Frozen model output ceiling sent to the pinned endpoint.",
    )
    run.add_argument(
        "--openrouter-output-limit-source",
        choices=["author_native", "benchmark_floor"],
        help="Evidence source for the frozen output ceiling.",
    )
    run.add_argument(
        "--openrouter-endpoint-cap-status",
        choices=["catalog_numeric", "undisclosed_first_party_exception"],
        help="Catalog evidence status for the selected endpoint completion cap.",
    )
    run.add_argument(
        "--openrouter-output-token-parameter",
        choices=["max_tokens", "max_completion_tokens"],
        help="Exact output-ceiling parameter exposed by the pinned endpoint.",
    )
    run.add_argument(
        "--openrouter-route-revision",
        help="Frozen route revision recorded in request identity and per-result provenance.",
    )
    run.add_argument(
        "--configuration-id",
        help="Frozen campaign configuration identity recorded in cache, results, and route provenance.",
    )
    run.add_argument(
        "--coda-api-key",
        help="Coda credential; falls back to CODA_API_KEY and then .env.",
    )
    run.add_argument("--coda-agents-url")
    run.add_argument("--coda-mode", choices=["build", "learn"], default="build")
    run.add_argument("--coda-fast", action="store_true")
    run.add_argument("--coda-prefer-structured-response", action="store_true")
    run.add_argument("--temperature", type=non_negative_float)
    reasoning = run.add_mutually_exclusive_group()
    reasoning.add_argument(
        "--reasoning-effort",
        choices=["max", "xhigh", "high", "medium", "low", "minimal", "none", ALL_REASONING_TOKEN],
        help="OpenRouter reasoning effort; use a level supported by the selected model.",
    )
    reasoning.add_argument(
        "--reasoning-enabled",
        action="store_true",
        default=None,
        help="Enable OpenRouter reasoning for models that do not expose effort levels.",
    )
    run.add_argument("--timeout", type=positive_float)
    run.add_argument("--max-retries", type=int, default=3)
    run.add_argument("--retry-base-delay", type=non_negative_float, default=1.0)
    run.add_argument("--retry-max-delay", type=positive_float, default=60.0)
    run.add_argument("--generation-concurrency", type=positive_int, default=1)
    run.add_argument("--evaluation-workers", type=positive_int, default=1)
    run.add_argument("--samples-per-task", type=positive_int, default=1)
    run.add_argument("--pass-k", type=positive_int, default=1)
    run.add_argument("--max-attempts", type=positive_int, default=1)
    run.add_argument("--feedback-max-chars", type=positive_int, default=2000)
    run.add_argument("--cache-dir", type=Path)
    run.add_argument("--resume-from", type=Path)
    run.add_argument("--task-timeout", type=positive_float)
    run.add_argument("--eval-timeout", type=positive_float, default=60.0)
    run.add_argument("--fail-fast", action="store_true")
    run.add_argument(
        "--stop-on-infrastructure-error",
        action="store_true",
        help="Drain the active generation chunk, then leave later prompts pending after an infrastructure failure.",
    )
    run.add_argument("--progress", action="store_true")


def _add_contract_arguments(contracts: argparse.ArgumentParser) -> None:
    subcommands = contracts.add_subparsers(dest="contracts_command", required=True)
    for name in ("validate", "list", "hash"):
        command = subcommands.add_parser(name, help=f"{name} semantic contracts")
        command.add_argument("--suite", default="core")
        command.add_argument("--path", type=Path, help="JSONL registry path; defaults to packaged suite")
    diff = subcommands.add_parser("diff", help="Diff two semantic contract registries")
    diff.add_argument("old", type=Path)
    diff.add_argument("new", type=Path)


def _validate_run_args(args: argparse.Namespace, run: argparse.ArgumentParser) -> None:
    _validate_matrix_args(args, run)
    _validate_phase_args(args, run)
    _validate_openrouter_args(args, run)
    _validate_coda_args(args, run)
    if args.max_retries < 0:
        run.error("--max-retries must be zero or greater")
    if args.tasks is not None and args.max_tasks is not None:
        run.error("--tasks cannot be combined with --max-tasks")
    if args.fail_fast and (args.samples_per_task > 1 or args.max_attempts > 1):
        run.error("fail-fast is incompatible with multi-sample or feedback modes.")
    if args.samples_per_task > 1 and args.temperature == 0.0:
        print(
            "Warning: all samples will be identical at temperature 0.0; consider --temperature 0.8 for Pass@K.",
            file=sys.stderr,
        )
    _validate_resume_args(args, run)


def _validate_matrix_args(args: argparse.Namespace, run: argparse.ArgumentParser) -> None:
    jobs = args.reasoning_jobs
    if args.registry and args.reasoning_enabled:
        run.error("--reasoning-enabled cannot be combined with --registry")
    if len(jobs) <= 1:
        return
    if args.provider == "coda":
        run.error("multi-job sweeps are not supported with --provider coda")
    if args.resume_from is not None:
        run.error("--resume-from cannot be combined with a multi-job sweep")
    if args.prompt is not None or args.regrade is not None:
        run.error("--rerun/--regrade cannot be combined with a multi-job sweep")
    if args.configuration_id is not None:
        run.error("--configuration-id is assigned automatically for multi-job sweeps")


def _validate_phase_args(args: argparse.Namespace, run: argparse.ArgumentParser) -> None:
    prompt_frameworks, regrade_frameworks = _phase_frameworks(args, run)
    if args.prompt is None and args.regrade is None:
        return
    _validate_phase_selection(args, run, prompt_frameworks, regrade_frameworks)
    _validate_phase_input(args, run, prompt_frameworks, regrade_frameworks)
    _validate_phase_incompatibilities(args, run)


def _phase_frameworks(
    args: argparse.Namespace, run: argparse.ArgumentParser
) -> tuple[tuple[Framework, ...], tuple[Framework, ...]]:
    try:
        prompt_frameworks = _frameworks(args.prompt) if args.prompt else ()
        regrade_frameworks = _frameworks(args.regrade) if args.regrade else ()
        _frameworks(args.framework)
    except ValueError as exc:
        run.error(str(exc))
    return prompt_frameworks, regrade_frameworks


def _validate_phase_selection(
    args: argparse.Namespace,
    run: argparse.ArgumentParser,
    prompt_frameworks: tuple[Framework, ...],
    regrade_frameworks: tuple[Framework, ...],
) -> None:
    if not prompt_frameworks and not regrade_frameworks:
        run.error("select at least one framework with --rerun or --regrade")
    if args.framework is None:
        return
    scope = set(_frameworks(args.framework))
    phases = set(prompt_frameworks) | set(regrade_frameworks)
    if not phases <= scope:
        run.error("--rerun and --regrade frameworks must be included in --framework when it is set")


def _validate_phase_input(
    args: argparse.Namespace,
    run: argparse.ArgumentParser,
    prompt_frameworks: tuple[Framework, ...],
    regrade_frameworks: tuple[Framework, ...],
) -> None:
    if any(framework not in prompt_frameworks for framework in regrade_frameworks) and args.input is None:
        run.error("--input is required for regrade-only frameworks")
    if args.input is None:
        return
    if not args.input.exists():
        run.error("--input must point to an existing JSONL output file")
    if args.input.suffix not in {".json", ".jsonl"}:
        run.error("--input only supports JSON or JSONL run output files")


def _validate_phase_incompatibilities(args: argparse.Namespace, run: argparse.ArgumentParser) -> None:
    if args.resume_from is not None:
        run.error("--resume-from cannot be combined with --rerun or --regrade")
    if args.max_attempts > 1:
        run.error("--rerun or --regrade cannot be combined with feedback mode")
    if args.fail_fast:
        run.error("--rerun or --regrade cannot be combined with --fail-fast")
    if args.task_timeout is not None:
        run.error("--rerun or --regrade cannot be combined with --task-timeout")


def _validate_openrouter_args(args: argparse.Namespace, run: argparse.ArgumentParser) -> None:
    route_pin_values = (
        args.openrouter_endpoint_tag,
        args.openrouter_max_output_tokens,
        args.openrouter_output_limit_source,
        args.openrouter_endpoint_cap_status,
        args.openrouter_output_token_parameter,
        args.openrouter_route_revision,
    )
    has_route_pin = any(value is not None for value in route_pin_values)
    if (has_route_pin or args.configuration_id is not None) and args.provider != "openrouter":
        run.error("OpenRouter endpoint pinning flags require --provider openrouter")
    if has_route_pin and not all(value is not None for value in route_pin_values):
        run.error(
            "--openrouter-endpoint-tag, --openrouter-max-output-tokens, --openrouter-output-limit-source, "
            "--openrouter-endpoint-cap-status, --openrouter-output-token-parameter, and "
            "--openrouter-route-revision must be supplied together"
        )
    if args.configuration_id is not None and not has_route_pin and not args.matrix_requested:
        run.error("--configuration-id requires a complete pinned OpenRouter endpoint route")
    if args.provider != "openrouter" or not _prompts_requested(args):
        return
    if args.openrouter_api_key_file is not None and not args.openrouter_api_key_file.is_file():
        run.error("--openrouter-api-key-file must point to a readable file")
    try:
        api_key = _openrouter_api_key(args)
    except (OSError, ValueError) as exc:
        run.error(str(exc))
    if not api_key:
        run.error(
            "OpenRouter credentials are required: use --openrouter-api-key, "
            "--openrouter-api-key-file, OPENROUTER_API_KEY, or .env"
        )
    if not args.model and not args.registry:
        run.error("--model is required when --provider openrouter")


def _validate_coda_args(args: argparse.Namespace, run: argparse.ArgumentParser) -> None:
    if args.provider != "coda":
        return
    if _prompts_requested(args):
        try:
            api_key = _coda_api_key(args)
        except (OSError, ValueError) as exc:
            run.error(str(exc))
        if not api_key:
            run.error("Coda credentials are required: use --coda-api-key, CODA_API_KEY, or .env")
    if args.model is None:
        args.model = _coda_model_label(args.coda_mode, args.coda_fast)
    if args.temperature is not None:
        print("Warning: coda api does not expose temperature; --temperature is ignored.", file=sys.stderr)


def _coda_model_label(mode: str, fast: bool) -> str:
    suffix = "-fast" if fast else ""
    return f"coda/{mode}{suffix}"


def _validate_resume_args(args: argparse.Namespace, run: argparse.ArgumentParser) -> None:
    if args.resume_from is None:
        return
    if not args.resume_from.exists():
        run.error("--resume-from must point to an existing JSONL output file")
    if args.resume_from.suffix != ".jsonl":
        run.error("--resume-from only supports JSONL output files")


def _run_command(args: argparse.Namespace) -> int:
    jobs: tuple[ReasoningJob, ...] = args.reasoning_jobs
    if not args.matrix_requested:
        return _run(args)
    if len(jobs) == 1:
        expanded = apply_reasoning_job(args, jobs[0], out=args.out, assign_configuration_id=True)
        return _run(expanded)

    entries: list[dict[str, Any]] = []
    multi_model = is_multi_model(jobs)
    exit_code = 0
    for job in jobs:
        out = sweep_output_path(args.out, job, multi_model=multi_model)
        expanded = apply_reasoning_job(args, job, out=out, assign_configuration_id=True)
        exit_code = _run(expanded)
        entries.append(
            {
                "model": job.model,
                "reasoning_effort": job.effort,
                "configuration_id": job.configuration_id(),
                "out": str(out),
                "exit_code": exit_code,
            }
        )
        if exit_code != 0:
            break
    manifest = sweep_manifest_path(args.out)
    write_sweep_manifest(manifest, entries)
    print(f"wrote {manifest}")
    return exit_code


def _run(args: argparse.Namespace) -> int:
    if args.prompt is None and args.regrade is None:
        prompt_frameworks = None
        regrade_frameworks = None
    else:
        prompt_frameworks = _frameworks(args.prompt) if args.prompt else ()
        regrade_frameworks = _frameworks(args.regrade) if args.regrade else ()
    frameworks = _selected_frameworks(args.framework, prompt_frameworks, regrade_frameworks)
    provider_config = _provider_config(args, include_credentials=_prompts_requested(args))
    try:
        config = RunConfig(
            provider=args.provider,
            frameworks=frameworks,
            source_hint=None if args.source_hint is None else Path(args.source_hint).expanduser(),
            model=args.model,
            max_tasks=args.max_tasks,
            task_numbers=None if args.tasks is None else tuple(args.tasks),
            provider_config={key: value for key, value in provider_config.items() if value is not None},
            suites=_suites(args.suite),
            samples_per_task=args.samples_per_task,
            pass_k=args.pass_k,
            max_attempts=args.max_attempts,
            feedback_max_chars=args.feedback_max_chars,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    provider = build_provider(config.provider, model=config.model, config=config.provider_config)
    options = RunOptions(
        generation_concurrency=args.generation_concurrency,
        evaluation_workers=args.evaluation_workers,
        cache_dir=None if args.cache_dir is None else args.cache_dir.expanduser(),
        resume_from=None if args.resume_from is None else args.resume_from.expanduser(),
        stream_to=_stream_path(args),
        # Provider request timeout and whole-task kill timeout are independent:
        # retries plus backoff may legitimately exceed one HTTP request.
        task_timeout=args.task_timeout,
        eval_timeout=args.eval_timeout,
        fail_fast=args.fail_fast,
        progress=args.progress,
        prompt_frameworks=prompt_frameworks,
        regrade_frameworks=regrade_frameworks,
        input_from=None if args.input is None else args.input.expanduser(),
        stop_on_infrastructure_error=args.stop_on_infrastructure_error,
    )
    try:
        payload = BenchmarkRunner(
            config=config,
            provider=provider,
            adapter=Adaptor(config.source_hint),
            options=options,
        ).run()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.configuration_id is not None:
        payload["configuration_id"] = args.configuration_id
    if options.stream_to is None or args.configuration_id is not None:
        write_output(args.out, payload, args.output_format)
    summary = payload["summary"]
    print(format_run_summary(summary))
    print(f"wrote {args.out}")
    return 0


def _contracts(args: argparse.Namespace) -> int:
    try:
        if args.contracts_command == "diff":
            old = ContractRegistry.from_path(args.old.expanduser())
            new = ContractRegistry.from_path(args.new.expanduser())
            for change in old.diff(new):
                print(
                    json.dumps(
                        {
                            "kind": change.kind,
                            "suite": change.suite,
                            "task_id": change.task_id,
                            "old_version": change.old_version,
                            "new_version": change.new_version,
                            "old_hash": change.old_hash,
                            "new_hash": change.new_hash,
                        },
                        sort_keys=True,
                    )
                )
            return 0
        registry = _contract_registry(args)
    except Exception as exc:  # noqa: BLE001 - CLI reports validation failures tersely.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.contracts_command == "validate":
        print(f"validated {len(registry)} contracts")
        return 0
    if args.contracts_command == "list":
        for contract in registry:
            print(
                json.dumps(
                    {
                        "suite": contract.suite,
                        "task_id": contract.task_id,
                        "kind": contract.kind.value,
                        "audit_status": contract.audit_status.value,
                        "shadow_only": contract.shadow_only,
                        "target": contract.target.target_id,
                    },
                    sort_keys=True,
                )
            )
        return 0
    if args.contracts_command == "hash":
        for contract in registry:
            print(
                json.dumps(
                    {
                        "suite": contract.suite,
                        "task_id": contract.task_id,
                        "contract_hash": contract_hash(contract),
                        "target_hash": contract.target.sha256,
                    },
                    sort_keys=True,
                )
            )
        return 0
    return 2


def _contract_registry(args: argparse.Namespace) -> ContractRegistry:
    if args.path is not None:
        return ContractRegistry.from_path(args.path.expanduser())
    return ContractRegistry.from_package(args.suite)


def _provider_config(args: argparse.Namespace, *, include_credentials: bool = True) -> dict[str, Any]:
    """Return provider-aware config for registry and cache identity.

    Args:
        args: Parsed command-line namespace.
        include_credentials: Resolve provider credentials when generation is
            part of the run. Regrade-only runs leave credentials out entirely.

    Returns:
        Provider config containing only keys used by the selected provider.
    """
    if args.provider == "smoke":
        config = {
            "smoke_mode": args.smoke_mode,
            "reasoning_effort": args.reasoning_effort,
            "reasoning_enabled": args.reasoning_enabled,
            "configuration_id": args.configuration_id,
        }
    elif args.provider == "openrouter":
        config = {
            "openrouter_api_key": _openrouter_api_key(args) if include_credentials else None,
            "openrouter_base_url": args.openrouter_base_url,
            "openrouter_endpoint_tag": args.openrouter_endpoint_tag,
            "openrouter_max_output_tokens": args.openrouter_max_output_tokens,
            "openrouter_output_limit_source": args.openrouter_output_limit_source,
            "openrouter_endpoint_cap_status": args.openrouter_endpoint_cap_status,
            "openrouter_output_token_parameter": args.openrouter_output_token_parameter,
            "openrouter_route_revision": args.openrouter_route_revision,
            # OpenRouter rejects configuration identity without a pinned route.
            "configuration_id": args.configuration_id if args.openrouter_endpoint_tag is not None else None,
            "temperature": args.temperature,
            "reasoning_effort": args.reasoning_effort,
            "reasoning_enabled": args.reasoning_enabled,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "retry_base_delay": args.retry_base_delay,
            "retry_max_delay": args.retry_max_delay,
        }
    else:
        config = {
            "coda_api_key": _coda_api_key(args) if include_credentials else None,
            "coda_agents_url": args.coda_agents_url,
            "coda_mode": args.coda_mode,
            "coda_fast": args.coda_fast,
            "coda_prefer_structured_response": args.coda_prefer_structured_response,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "retry_base_delay": args.retry_base_delay,
            "retry_max_delay": args.retry_max_delay,
        }
    return {key: value for key, value in config.items() if value is not None}


def _openrouter_api_key(args: argparse.Namespace) -> str | None:
    if args.openrouter_api_key:
        return str(args.openrouter_api_key).strip() or None
    if args.openrouter_api_key_file is not None:
        value = args.openrouter_api_key_file.expanduser().read_text(encoding="utf-8").strip()
        return value or None
    return _environment_or_dotenv("OPENROUTER_API_KEY")


def _coda_api_key(args: argparse.Namespace) -> str | None:
    if args.coda_api_key:
        return str(args.coda_api_key).strip() or None
    return _environment_or_dotenv("CODA_API_KEY")


def _environment_or_dotenv(name: str) -> str | None:
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    return _dotenv_value(Path.cwd() / ".env", name)


def _dotenv_value(path: Path, name: str) -> str | None:
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, raw_value = line.partition("=")
        if separator and key.strip() == name:
            return _parse_dotenv_value(raw_value, name)
    return None


def _parse_dotenv_value(raw_value: str, name: str) -> str | None:
    value = raw_value.strip()
    if not value:
        return None
    if value[0] not in {'"', "'"}:
        return value.split(" #", 1)[0].strip() or None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid quoted value for {name} in .env") from exc
    if not isinstance(parsed, str):
        raise ValueError(f"{name} in .env must be a string")
    return parsed.strip() or None


def _frameworks(choice: str | Sequence[Sequence[str]] | Sequence[str] | None) -> tuple[Framework, ...]:
    """Normalize one or more framework choices while preserving their order."""
    choices = _flatten_choices(choice)
    if not choices or "all" in choices:
        if len(set(choices)) > 1 and "all" in choices:
            raise ValueError("all cannot be combined with named frameworks")
        return DEFAULT_FRAMEWORKS
    return tuple(cast(Framework, item) for item in dict.fromkeys(choices))


def _selected_frameworks(
    framework_choices: str | Sequence[Sequence[str]] | Sequence[str] | None,
    prompt_frameworks: tuple[Framework, ...] | None,
    regrade_frameworks: tuple[Framework, ...] | None,
) -> tuple[Framework, ...]:
    """Return the framework union selected by the phase and scope flags."""
    if prompt_frameworks is None and regrade_frameworks is None:
        return _frameworks(framework_choices)
    selected: list[Framework] = []
    for group in (prompt_frameworks or (), regrade_frameworks or ()):
        selected.extend(group)
    return tuple(dict.fromkeys(selected))


def _flatten_choices(choice: str | Sequence[Sequence[str]] | Sequence[str] | None) -> list[str]:
    if choice is None:
        return []
    if isinstance(choice, str):
        return [choice]
    flattened: list[str] = []
    for item in choice:
        if isinstance(item, str):
            flattened.append(item)
        else:
            flattened.extend(item)
    return flattened


def _prompts_requested(args: argparse.Namespace) -> bool:
    if args.prompt is None and args.regrade is None:
        return True
    return bool(args.prompt)


def _suites(choice: str) -> tuple[Suite, ...]:
    if choice == "all":
        return ("core", "qec")
    return (choice,)  # type: ignore[return-value]


def _stream_path(args: argparse.Namespace) -> Path | None:
    if infer_format(args.out, args.output_format) != "jsonl":
        return None
    if (
        args.generation_concurrency > 1
        or args.evaluation_workers > 1
        or args.resume_from is not None
        or args.task_timeout is not None
        or args.timeout is not None
        or args.samples_per_task > 1
        or args.max_attempts > 1
    ):
        return args.out
    return None
