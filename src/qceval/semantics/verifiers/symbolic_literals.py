"""Certified numeric-literal policy for bounded symbolic verification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import StrEnum
from fractions import Fraction

_PI_DECIMAL = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286")
_MAX_PI_DENOMINATOR = 64
_MAX_EXACT_BINARY_DENOMINATOR = 64
_ULP_MULTIPLIER = 8


class LiteralKind(StrEnum):
    """Outcome of conservative floating-literal certification."""

    EXACT_RATIONAL = "exact_rational"
    PI_MULTIPLE = "pi_multiple"
    UNMATCHED = "unmatched"


@dataclass(frozen=True)
class LiteralCertification:
    """Exact replacement metadata and conservative absolute error."""

    kind: LiteralKind
    numerator: int | None
    denominator: int | None
    absolute_error: float | None


def certify_float(value: float) -> LiteralCertification:
    """Certify a float as an exact small rational or near-pi multiple.

    Args:
        value: Finite candidate floating value.

    Returns:
        Replacement metadata. Unmatched values have no exact replacement.
    """
    if not math.isfinite(value):
        return LiteralCertification(LiteralKind.UNMATCHED, None, None, None)
    exact = Fraction.from_float(value)
    if exact.denominator <= _MAX_EXACT_BINARY_DENOMINATOR:
        return LiteralCertification(LiteralKind.EXACT_RATIONAL, exact.numerator, exact.denominator, 0.0)
    ratio = Fraction(value / math.pi).limit_denominator(_MAX_PI_DENOMINATOR)
    with localcontext() as context:
        context.prec = 90
        observed = Decimal.from_float(value)
        ideal = Decimal(ratio.numerator) * _PI_DECIMAL / Decimal(ratio.denominator)
        error = abs(observed - ideal)
        threshold = Decimal.from_float(math.ulp(value)) * _ULP_MULTIPLIER
    if error <= threshold:
        return LiteralCertification(
            LiteralKind.PI_MULTIPLE,
            ratio.numerator,
            ratio.denominator,
            float(error),
        )
    return LiteralCertification(LiteralKind.UNMATCHED, None, None, None)
