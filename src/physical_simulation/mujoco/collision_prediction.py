"""Analytic collision prediction helpers for MuJoCo substep planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from physical_simulation.mujoco.contact_params import MuJoCoContactSolverParams
from physical_simulation.mujoco.contact_timescale import (
    SolverContactTimescale,
    SubstepRecommendation,
    SubstepRecommendationConfig,
    estimate_solver_contact_timescale,
    recommend_solver_substeps,
)
from physical_simulation.validation.asset_validator import _as_float_tuple, _finite_float
from physical_simulation.validation.errors import PhysicsValidationError

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class AnalyticPlane:
    """Infinite analytic plane used for constant-velocity collision prediction."""

    point: Vector3
    normal: Vector3
    linear_velocity: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _vector3(self.point, field_name="point"))
        normal = _vector3(self.normal, field_name="normal")
        normal_length = _norm(normal)
        if normal_length <= 1.0e-12:
            raise PhysicsValidationError(f"normal must have non-zero length; actual value={self.normal!r}")
        object.__setattr__(self, "normal", _scale(normal, 1.0 / normal_length))
        object.__setattr__(self, "linear_velocity", _vector3(self.linear_velocity, field_name="linear_velocity"))


@dataclass(frozen=True)
class CollisionPrediction:
    """Predicted first contact event for simple analytic primitives."""

    collision_type: str
    time_to_contact: float
    gap: float
    normal_approach_speed: float
    contact_normal: Vector3

    def __post_init__(self) -> None:
        if not isinstance(self.collision_type, str) or not self.collision_type.strip():
            raise PhysicsValidationError(f"collision_type must be a non-empty string; actual value={self.collision_type!r}")
        object.__setattr__(
            self,
            "time_to_contact",
            _finite_float(self.time_to_contact, field_name="time_to_contact", minimum=0.0, error_type=PhysicsValidationError),
        )
        object.__setattr__(self, "gap", _finite_float(self.gap, field_name="gap", error_type=PhysicsValidationError))
        object.__setattr__(
            self,
            "normal_approach_speed",
            _finite_float(self.normal_approach_speed, field_name="normal_approach_speed", minimum=0.0, error_type=PhysicsValidationError),
        )
        normal = _vector3(self.contact_normal, field_name="contact_normal")
        normal_length = _norm(normal)
        if normal_length <= 1.0e-12:
            raise PhysicsValidationError(f"contact_normal must have non-zero length; actual value={self.contact_normal!r}")
        object.__setattr__(self, "contact_normal", _scale(normal, 1.0 / normal_length))


@dataclass(frozen=True)
class SolverCollisionEstimate:
    """Collision prediction paired with solver timescale and substep recommendation."""

    prediction: CollisionPrediction
    timescale: SolverContactTimescale
    recommendation: SubstepRecommendation


def predict_sphere_plane_collision(
    *,
    sphere_position: Vector3,
    sphere_velocity: Vector3,
    sphere_radius: float,
    plane: AnalyticPlane,
    prediction_horizon: float,
) -> CollisionPrediction | None:
    """Predict first constant-velocity collision between a sphere and an analytic plane."""
    center = _vector3(sphere_position, field_name="sphere_position")
    velocity = _vector3(sphere_velocity, field_name="sphere_velocity")
    radius = _finite_float(sphere_radius, field_name="sphere_radius", minimum=0.0, strict_minimum=True, error_type=PhysicsValidationError)
    horizon = _finite_float(
        prediction_horizon,
        field_name="prediction_horizon",
        minimum=0.0,
        strict_minimum=True,
        error_type=PhysicsValidationError,
    )
    gap = _dot(_subtract(center, plane.point), plane.normal) - radius
    relative_normal_velocity = _dot(_subtract(velocity, plane.linear_velocity), plane.normal)
    if gap <= 0.0:
        return CollisionPrediction("sphere_plane", 0.0, gap, max(0.0, -relative_normal_velocity), plane.normal)
    if relative_normal_velocity >= 0.0:
        return None
    time_to_contact = gap / -relative_normal_velocity
    if time_to_contact > horizon:
        return None
    return CollisionPrediction("sphere_plane", time_to_contact, gap, -relative_normal_velocity, plane.normal)


def predict_sphere_sphere_collision(
    *,
    sphere_a_position: Vector3,
    sphere_a_velocity: Vector3,
    sphere_a_radius: float,
    sphere_b_position: Vector3,
    sphere_b_velocity: Vector3,
    sphere_b_radius: float,
    prediction_horizon: float,
) -> CollisionPrediction | None:
    """Predict first constant-velocity collision between two spheres."""
    c1 = _vector3(sphere_a_position, field_name="sphere_a_position")
    c2 = _vector3(sphere_b_position, field_name="sphere_b_position")
    v1 = _vector3(sphere_a_velocity, field_name="sphere_a_velocity")
    v2 = _vector3(sphere_b_velocity, field_name="sphere_b_velocity")
    r1 = _finite_float(sphere_a_radius, field_name="sphere_a_radius", minimum=0.0, strict_minimum=True, error_type=PhysicsValidationError)
    r2 = _finite_float(sphere_b_radius, field_name="sphere_b_radius", minimum=0.0, strict_minimum=True, error_type=PhysicsValidationError)
    horizon = _finite_float(
        prediction_horizon,
        field_name="prediction_horizon",
        minimum=0.0,
        strict_minimum=True,
        error_type=PhysicsValidationError,
    )
    r = _subtract(c2, c1)
    v = _subtract(v2, v1)
    radius_sum = r1 + r2
    distance = _norm(r)
    gap = distance - radius_sum
    if gap <= 0.0:
        normal = _safe_normal(r, v)
        approach_speed = max(0.0, -_dot(v, normal))
        return CollisionPrediction("sphere_sphere", 0.0, gap, approach_speed, normal)
    a = _dot(v, v)
    if a <= 1.0e-24:
        return None
    b = 2.0 * _dot(r, v)
    c = _dot(r, r) - radius_sum * radius_sum
    discriminant = b * b - 4.0 * a * c
    if discriminant < -1.0e-12:
        return None
    discriminant = max(0.0, discriminant)
    root = math.sqrt(discriminant)
    candidates = tuple(t for t in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if t >= -1.0e-12)
    if not candidates:
        return None
    time_to_contact = max(0.0, min(candidates))
    if time_to_contact > horizon:
        return None
    normal = _safe_normal(_add(r, _scale(v, time_to_contact)), v)
    approach_speed = max(0.0, -_dot(v, normal))
    return CollisionPrediction("sphere_sphere", time_to_contact, gap, approach_speed, normal)


def estimate_solver_collision(
    *,
    prediction: CollisionPrediction,
    params: MuJoCoContactSolverParams,
    macro_timestep: float,
    config: SubstepRecommendationConfig | None = None,
) -> SolverCollisionEstimate:
    """Combine a geometry prediction with a solver-timescale substep recommendation."""
    timescale = estimate_solver_contact_timescale(params)
    recommendation = recommend_solver_substeps(
        macro_timestep=macro_timestep,
        timescale=timescale,
        params=params,
        config=config,
    )
    return SolverCollisionEstimate(
        prediction=prediction,
        timescale=timescale,
        recommendation=recommendation,
    )


def _vector3(value: Any, *, field_name: str) -> Vector3:
    return tuple(_as_float_tuple(value, field_name=field_name, length=3, error_type=PhysicsValidationError))  # type: ignore[return-value]


def _add(first: Vector3, second: Vector3) -> Vector3:
    return tuple(first[index] + second[index] for index in range(3))  # type: ignore[return-value]


def _subtract(first: Vector3, second: Vector3) -> Vector3:
    return tuple(first[index] - second[index] for index in range(3))  # type: ignore[return-value]


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return tuple(value * scalar for value in vector)  # type: ignore[return-value]


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(first[index] * second[index] for index in range(3))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _safe_normal(offset: Vector3, fallback_velocity: Vector3) -> Vector3:
    offset_norm = _norm(offset)
    if offset_norm > 1.0e-12:
        return _scale(offset, 1.0 / offset_norm)
    velocity_norm = _norm(fallback_velocity)
    if velocity_norm > 1.0e-12:
        return _scale(fallback_velocity, 1.0 / velocity_norm)
    return (1.0, 0.0, 0.0)
