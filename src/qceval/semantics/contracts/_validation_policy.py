"""Policy, target, routing, and limit parsers for contracts."""

from __future__ import annotations

from typing import Any

from qceval.semantics.contracts._validation_primitives import (
    SHA256_PATTERN,
    array_value,
    boolean_value,
    enum_value,
    fail,
    freeze_json,
    integer_value,
    nonempty_string,
    number_tuple,
    number_value,
    object_value,
    parameter_point_tuple,
    positive_number,
    semantic_version,
    string_tuple,
    string_value,
    unique,
)
from qceval.semantics.contracts.kinds import (
    ApproximationMode,
    ApproximationSpec,
    DiagnosticSpec,
    LimitsSpec,
    ParameterQuantifier,
    ParameterSpec,
    ParametersSpec,
    RequirementSpec,
    RouteSpec,
    RoutingSpec,
    TargetSpec,
)


def parse_parameters(value: Any) -> ParametersSpec:
    """Parse parameter domains and quantification.

    Args:
        value: Untrusted JSON-compatible parameters value.

    Returns:
        Validated parameter-family specification.

    Raises:
        ContractValidationError: If parameters or diagnostic points are
            malformed.
    """
    raw = object_value(
        value,
        "$.parameters",
        required={"items", "quantifier", "completeness", "diagnostic_points"},
    )
    items = tuple(
        _parse_parameter(item, f"$.parameters.items[{index}]")
        for index, item in enumerate(array_value(raw["items"], "$.parameters.items"))
    )
    unique([item.name for item in items], "$.parameters.items", "parameter names")
    points = tuple(
        parameter_point_tuple(point, f"$.parameters.diagnostic_points[{index}]")
        for index, point in enumerate(array_value(raw["diagnostic_points"], "$.parameters.diagnostic_points"))
    )
    completeness = raw["completeness"]
    if completeness is not None:
        completeness = nonempty_string(completeness, "$.parameters.completeness")
    return ParametersSpec(
        items=items,
        quantifier=enum_value(
            ParameterQuantifier,
            raw["quantifier"],
            "$.parameters.quantifier",
        ),
        completeness=completeness,
        diagnostic_points=points,
    )


def _parse_parameter(value: Any, path: str) -> ParameterSpec:
    raw = object_value(
        value,
        path,
        required={"name", "type", "domain", "units", "periodicity", "excluded", "binding"},
    )
    periodicity = raw["periodicity"]
    if periodicity is not None:
        periodicity = number_value(periodicity, f"{path}.periodicity", minimum=0.0)
        if periodicity <= 0.0:
            fail(f"{path}.periodicity", "must be positive")
    return ParameterSpec(
        name=nonempty_string(raw["name"], f"{path}.name"),
        value_type=nonempty_string(raw["type"], f"{path}.type"),
        domain=nonempty_string(raw["domain"], f"{path}.domain"),
        units=nonempty_string(raw["units"], f"{path}.units"),
        periodicity=periodicity,
        excluded=number_tuple(raw["excluded"], f"{path}.excluded"),
        binding=nonempty_string(raw["binding"], f"{path}.binding"),
    )


def parse_approximation(value: Any) -> ApproximationSpec:
    """Parse approximation policy and error bounds.

    Args:
        value: Untrusted JSON-compatible approximation value.

    Returns:
        Validated approximation specification.

    Raises:
        ContractValidationError: If a mode, metric, or bound is malformed.
    """
    raw = object_value(
        value,
        "$.approximation",
        required={"mode", "metric", "tolerance", "uncertainty", "error_budget"},
    )
    return ApproximationSpec(
        mode=enum_value(ApproximationMode, raw["mode"], "$.approximation.mode"),
        metric=nonempty_string(raw["metric"], "$.approximation.metric"),
        tolerance=number_value(raw["tolerance"], "$.approximation.tolerance", minimum=0.0),
        uncertainty=number_value(raw["uncertainty"], "$.approximation.uncertainty", minimum=0.0),
        error_budget=number_value(raw["error_budget"], "$.approximation.error_budget", minimum=0.0),
    )


def parse_target(value: Any) -> TargetSpec:
    """Parse target identity and provenance.

    Args:
        value: Untrusted JSON-compatible target value.

    Returns:
        Validated target artifact specification.

    Raises:
        ContractValidationError: If target identity, provenance, or digest is
            malformed.
    """
    raw = object_value(
        value,
        "$.target",
        required={"id", "version", "sha256", "source", "manifest", "independent_derivations"},
    )
    digest = string_value(raw["sha256"], "$.target.sha256")
    if not SHA256_PATTERN.fullmatch(digest):
        fail("$.target.sha256", "must be 64 lowercase hexadecimal characters")
    return TargetSpec(
        target_id=nonempty_string(raw["id"], "$.target.id"),
        version=semantic_version(raw["version"], "$.target.version"),
        sha256=digest,
        source=nonempty_string(raw["source"], "$.target.source"),
        manifest=nonempty_string(raw["manifest"], "$.target.manifest"),
        independent_derivations=integer_value(
            raw["independent_derivations"],
            "$.target.independent_derivations",
            minimum=1,
        ),
    )


def parse_routing(value: Any) -> RoutingSpec:
    """Parse primary and fallback verifier routes.

    Args:
        value: Untrusted JSON-compatible routing value.

    Returns:
        Validated ordered verifier routes.

    Raises:
        ContractValidationError: If routes are absent, malformed, or
            duplicated.
    """
    raw = object_value(value, "$.routing", required={"primary", "fallback"})
    primary = _parse_routes(raw["primary"], "$.routing.primary")
    fallback = _parse_routes(raw["fallback"], "$.routing.fallback")
    if not primary:
        fail("$.routing.primary", "must contain at least one route")
    unique([route.engine for route in (*primary, *fallback)], "$.routing", "engine names")
    return RoutingSpec(primary=primary, fallback=fallback)


def _parse_routes(value: Any, path: str) -> tuple[RouteSpec, ...]:
    return tuple(_parse_route(item, f"{path}[{index}]") for index, item in enumerate(array_value(value, path)))


def _parse_route(value: Any, path: str) -> RouteSpec:
    raw = object_value(value, path, required={"engine", "capabilities", "cross_check"})
    return RouteSpec(
        engine=nonempty_string(raw["engine"], f"{path}.engine"),
        capabilities=string_tuple(raw["capabilities"], f"{path}.capabilities"),
        cross_check=boolean_value(raw["cross_check"], f"{path}.cross_check"),
    )


def parse_limits(value: Any) -> LimitsSpec:
    """Parse resource and complexity limits.

    Args:
        value: Untrusted JSON-compatible limits value.

    Returns:
        Validated resource and materialization limits.

    Raises:
        ContractValidationError: If any limit is missing or out of range.
    """
    raw = object_value(
        value,
        "$.limits",
        required={
            "wall_seconds",
            "cpu_seconds",
            "memory_mib",
            "max_qubits",
            "max_dimension",
            "max_cases",
            "max_branches",
            "max_expression_nodes",
        },
    )
    return LimitsSpec(
        wall_seconds=positive_number(raw["wall_seconds"], "$.limits.wall_seconds"),
        cpu_seconds=positive_number(raw["cpu_seconds"], "$.limits.cpu_seconds"),
        memory_mib=integer_value(raw["memory_mib"], "$.limits.memory_mib", minimum=1),
        max_qubits=integer_value(raw["max_qubits"], "$.limits.max_qubits", minimum=0),
        max_dimension=integer_value(raw["max_dimension"], "$.limits.max_dimension", minimum=1),
        max_cases=integer_value(raw["max_cases"], "$.limits.max_cases", minimum=1),
        max_branches=integer_value(raw["max_branches"], "$.limits.max_branches", minimum=1),
        max_expression_nodes=integer_value(
            raw["max_expression_nodes"],
            "$.limits.max_expression_nodes",
            minimum=1,
        ),
    )


def parse_requirements(value: Any) -> tuple[RequirementSpec, ...]:
    """Parse semantic requirements.

    Args:
        value: Untrusted JSON-compatible requirement array.

    Returns:
        Validated immutable requirement declarations.

    Raises:
        ContractValidationError: If requirements are malformed or duplicated.
    """
    items = tuple(
        _parse_requirement(item, f"$.requirements[{index}]")
        for index, item in enumerate(array_value(value, "$.requirements"))
    )
    unique([item.requirement_id for item in items], "$.requirements", "requirement ids")
    return items


def _parse_requirement(value: Any, path: str) -> RequirementSpec:
    raw = object_value(value, path, required={"id", "kind", "source", "value"})
    return RequirementSpec(
        requirement_id=nonempty_string(raw["id"], f"{path}.id"),
        kind=nonempty_string(raw["kind"], f"{path}.kind"),
        source=nonempty_string(raw["source"], f"{path}.source"),
        value=freeze_json(raw["value"], f"{path}.value"),
    )


def parse_diagnostics(value: Any) -> tuple[DiagnosticSpec, ...]:
    """Parse optional diagnostic declarations.

    Args:
        value: Untrusted JSON-compatible diagnostic array.

    Returns:
        Validated immutable diagnostic declarations.

    Raises:
        ContractValidationError: If diagnostics are malformed or duplicated.
    """
    items = tuple(
        _parse_diagnostic(item, f"$.diagnostics[{index}]")
        for index, item in enumerate(array_value(value, "$.diagnostics"))
    )
    unique([item.diagnostic_id for item in items], "$.diagnostics", "diagnostic ids")
    return items


def _parse_diagnostic(value: Any, path: str) -> DiagnosticSpec:
    raw = object_value(value, path, required={"id", "kind", "enabled"})
    return DiagnosticSpec(
        diagnostic_id=nonempty_string(raw["id"], f"{path}.id"),
        kind=nonempty_string(raw["kind"], f"{path}.kind"),
        enabled=boolean_value(raw["enabled"], f"{path}.enabled"),
    )
