"""Argument parser value converters."""

from __future__ import annotations

import argparse


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse.

    Args:
        value: Raw command-line value.

    Returns:
        Parsed integer greater than zero.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not greater than zero.
        ValueError: If ``value`` is not an integer string.
    """
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    """Parse a positive float for argparse.

    Args:
        value: Raw command-line value.

    Returns:
        Parsed float greater than zero.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not greater than zero.
        ValueError: If ``value`` is not a float string.
    """
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_float(value: str) -> float:
    """Parse a non-negative float for argparse.

    Args:
        value: Raw command-line value.

    Returns:
        Parsed float greater than or equal to zero.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is negative.
        ValueError: If ``value`` is not a float string.
    """
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed
