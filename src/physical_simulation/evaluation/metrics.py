"""Small numerical helpers for evaluation metrics."""

from __future__ import annotations

import math


def vector_norm(value: tuple[float, float, float]) -> float:
    """Return Euclidean norm for a 3D vector."""
    return math.sqrt(sum(float(component) * float(component) for component in value))


def quaternion_angular_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Return shortest angular distance between two quaternions in radians."""
    dot = sum(float(a) * float(b) for a, b in zip(first, second))
    clamped = max(0.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(clamped)
