"""Compatibility facade for exact distribution verification."""

from qceval.semantics.verifiers.distribution_engine import (
    DISTRIBUTION_ENGINE_VERSION,
    DistributionEngine,
    DistributionTargetProvider,
    PackagedDistributionTargetProvider,
)
from qceval.semantics.verifiers.distribution_materializers import (
    AdaptiveDistributionMaterializer,
    DistributionMaterializer,
    ExecutionDistributionMaterializer,
    ProbabilityTable,
    ProgramDistributionMaterializer,
)

__all__ = [
    "DISTRIBUTION_ENGINE_VERSION",
    "AdaptiveDistributionMaterializer",
    "DistributionEngine",
    "DistributionMaterializer",
    "DistributionTargetProvider",
    "ExecutionDistributionMaterializer",
    "PackagedDistributionTargetProvider",
    "ProbabilityTable",
    "ProgramDistributionMaterializer",
]
