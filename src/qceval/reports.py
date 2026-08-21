"""Compatibility facade for benchmark report generation."""

from __future__ import annotations

from qceval.reporting.aggregate import summarize
from qceval.reporting.costs import cost_summary
from qceval.reporting.error_taxonomy import error_taxonomy_summary
from qceval.reporting.feedback_lineage import feedback_lineage_summary
from qceval.reporting.formatting import format_run_summary
from qceval.reporting.protocol import pass_at_k
from qceval.reporting.semantic import semantic_summary

__all__ = [
    "cost_summary",
    "error_taxonomy_summary",
    "feedback_lineage_summary",
    "format_run_summary",
    "pass_at_k",
    "semantic_summary",
    "summarize",
]
