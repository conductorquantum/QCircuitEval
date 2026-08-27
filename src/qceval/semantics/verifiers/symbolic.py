"""Resource-bounded symbolic parameter-family verification."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import psutil

from qceval.semantics.contracts import contract_hash
from qceval.semantics.integration import SemanticVerificationRequest
from qceval.semantics.ir import source_code_sha256
from qceval.semantics.verifiers.result import (
    SemanticStatus,
    VerifierResult,
    make_evidence,
    make_verifier_result,
)

SYMBOLIC_ENGINE_VERSION = "1.0.0"
SYMBOLIC_COMPLETENESS = "bounded_symbolic_projective_identity_with_certified_literals"


@dataclass(frozen=True)
class SymbolicBudget:
    """Deterministic worker and expression limits."""

    wall_seconds: float
    cpu_seconds: int
    memory_mib: int
    max_expression_nodes: int
    max_expanded_nodes: int


@dataclass(frozen=True)
class SymbolicProof:
    """Bounded worker proof/refutation outcome."""

    outcome: str
    reason: str
    certified_error_bound: float | None
    gate_families: tuple[str, ...]
    peak_expression_nodes: int
    residuals: tuple[str, ...]
    elapsed_seconds: float


class BoundedSymbolicSourceVerifier:
    """Verify restricted continuous families from candidate source."""

    def verify(self, request: SemanticVerificationRequest) -> VerifierResult:
        """Run the contract-selected family proof in a bounded worker.

        Args:
            request: Evaluator semantic request with candidate source.

        Returns:
            Decisive, execution-error, or resource-limit result.
        """
        contract = request.contract
        routes_symbolic = any(
            route.engine == "symbolic_family_bounded"
            for route in (*contract.routing.primary, *contract.routing.fallback)
        )
        if contract.parameters.completeness != SYMBOLIC_COMPLETENESS and not routes_symbolic:
            return _result(request, SemanticStatus.EXECUTION_ERROR, "symbolic_completeness_unsupported", None)
        if request.code is None:
            return _result(request, SemanticStatus.EXECUTION_ERROR, "symbolic_source_unavailable", None)
        limits = contract.limits
        budget = SymbolicBudget(
            wall_seconds=limits.wall_seconds,
            cpu_seconds=max(1, math.ceil(limits.cpu_seconds)),
            memory_mib=limits.memory_mib,
            max_expression_nodes=limits.max_expression_nodes,
            max_expanded_nodes=limits.max_expression_nodes * 4,
        )
        proof = prove_projective_family(
            request.code,
            contract.signature.entry_point,
            framework=request.framework,
            tolerance=contract.approximation.tolerance,
            budget=budget,
        )
        status = _proof_status(proof)
        return _result(request, status, proof.reason, proof)


def _proof_status(proof: SymbolicProof) -> SemanticStatus:
    """Map a bounded proof outcome to a decisive or operational status."""
    if proof.outcome == "proved":
        return SemanticStatus.VERIFIED_PASS
    if proof.outcome == "refuted":
        return SemanticStatus.SEMANTIC_FAIL
    resource_markers = (
        "resource_limit",
        "timeout",
        "memory_limit",
        "expression_node_limit",
        "expansion_node_limit",
    )
    if any(marker in proof.reason for marker in resource_markers):
        return SemanticStatus.RESOURCE_LIMIT
    return SemanticStatus.EXECUTION_ERROR


def prove_projective_family(
    code: str,
    entry_point: str,
    *,
    framework: str = "qiskit",
    tolerance: float,
    budget: SymbolicBudget,
) -> SymbolicProof:
    """Prove or refute a restricted family in an isolated worker.

    Args:
        code: Candidate Python source.
        entry_point: Candidate builder name.
        framework: Source framework syntax.
        tolerance: Contract semantic tolerance.
        budget: Worker and expression limits.

    Returns:
        Bounded proof, refutation, or inconclusive outcome.
    """
    payload = json.dumps(
        {
            "code": code,
            "entry_point": entry_point,
            "framework": framework,
            "tolerance": tolerance,
            "max_expression_nodes": budget.max_expression_nodes,
            "max_expanded_nodes": budget.max_expanded_nodes,
            "cpu_seconds": budget.cpu_seconds,
            "memory_mib": budget.memory_mib,
        }
    )
    started = time.perf_counter()
    worker = _run_worker(payload, budget, started)
    if isinstance(worker, SymbolicProof):
        return worker
    returncode, stdout = worker
    elapsed = time.perf_counter() - started
    if returncode != 0:
        reason = "symbolic_worker_resource_limit" if returncode < 0 else "symbolic_worker_failure"
        return _proof("inconclusive", reason, elapsed)
    if len(stdout.encode("utf-8")) > 100_000:
        return _proof("inconclusive", "symbolic_worker_output_limit", elapsed)
    try:
        value = json.loads(stdout)
        return _parse_proof(value, elapsed)
    except (TypeError, ValueError, json.JSONDecodeError):
        return _proof("inconclusive", "symbolic_worker_protocol_error", elapsed)


def _run_worker(
    payload: str,
    budget: SymbolicBudget,
    started: float,
) -> tuple[int, str] | SymbolicProof:
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "qceval.frameworks.qiskit.symbolic"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return _proof("inconclusive", "symbolic_worker_start_failure", time.perf_counter() - started)
    assert process.stdin is not None
    try:
        process.stdin.write(payload)
        process.stdin.close()
    except (BrokenPipeError, OSError):
        _stop_worker(process)
        return _proof("inconclusive", "symbolic_worker_protocol_error", time.perf_counter() - started)
    try:
        monitored = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        process.wait()
        assert process.stdout is not None
        stdout = process.stdout.read()
        process.stdout.close()
        return process.returncode, stdout
    memory_limit = budget.memory_mib * 1024 * 1024
    while process.poll() is None:
        elapsed = time.perf_counter() - started
        if elapsed > budget.wall_seconds:
            _stop_worker(process)
            return _proof("inconclusive", "symbolic_worker_timeout", elapsed)
        try:
            resident = monitored.memory_info().rss
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            resident = 0
        if resident > memory_limit:
            _stop_worker(process)
            return _proof("inconclusive", "symbolic_worker_memory_limit", elapsed)
        time.sleep(0.01)
    assert process.stdout is not None
    stdout = process.stdout.read()
    process.stdout.close()
    return process.returncode, stdout


def _stop_worker(process: subprocess.Popen[str]) -> None:
    process.kill()
    process.wait()
    if process.stdout is not None:
        process.stdout.close()


def _parse_proof(value: Any, elapsed: float) -> SymbolicProof:
    if not isinstance(value, dict) or value.get("outcome") not in {"proved", "refuted", "inconclusive"}:
        raise ValueError("invalid symbolic worker result")
    reason = value.get("reason")
    gates = value.get("gate_families")
    residuals = value.get("residuals")
    peak = value.get("peak_expression_nodes")
    error = value.get("certified_error_bound")
    if (
        not isinstance(reason, str)
        or not reason
        or not isinstance(gates, list)
        or not all(isinstance(item, str) for item in gates)
        or not isinstance(residuals, list)
        or not all(isinstance(item, str) for item in residuals)
        or not isinstance(peak, int)
        or peak < 0
        or (error is not None and (not isinstance(error, int | float) or not math.isfinite(error) or error < 0))
    ):
        raise ValueError("invalid symbolic worker fields")
    return SymbolicProof(
        str(value["outcome"]),
        reason[:500],
        None if error is None else float(error),
        tuple(gates[:64]),
        peak,
        tuple(item[:300] for item in residuals[:8]),
        elapsed,
    )


def _proof(outcome: str, reason: str, elapsed: float) -> SymbolicProof:
    return SymbolicProof(outcome, reason, None, (), 0, (), elapsed)


def _result(
    request: SemanticVerificationRequest,
    status: SemanticStatus,
    reason: str,
    proof: SymbolicProof | None,
) -> VerifierResult:
    contract = request.contract
    error = None if proof is None else proof.certified_error_bound
    input_hash = source_code_sha256(request.code)
    evidence = make_evidence(
        "symbolic_family_bounded",
        SYMBOLIC_ENGINE_VERSION,
        reason,
        input_hash=input_hash,
        target_hash=contract.target.sha256,
        metric="certified_literal_error" if error is not None else None,
        value=error,
        tolerance=contract.approximation.tolerance if error is not None else None,
        uncertainty=contract.approximation.uncertainty if error is not None else None,
        elapsed_seconds=0.0 if proof is None else proof.elapsed_seconds,
        preconditions=()
        if proof is None
        else (
            f"outcome={proof.outcome}",
            f"peak_expression_nodes={proof.peak_expression_nodes}",
            f"gate_families={','.join(proof.gate_families)}",
        ),
    )
    return make_verifier_result(
        status,
        reason,
        contract_hash=contract_hash(contract),
        target_hash=contract.target.sha256,
        verifier_version=SYMBOLIC_ENGINE_VERSION,
        evidence=(evidence,),
        diagnostics=()
        if proof is None
        else tuple((f"residual_{index}", value) for index, value in enumerate(proof.residuals)),
    )
