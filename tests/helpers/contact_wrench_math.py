"""Test contact wrench math helpers."""

from __future__ import annotations

import math

from physical_simulation.runtime import (
    ContactWrench,
    aggregate_contact_wrenches_by_body,
    force_on_body,
    pure_contact_torque_on_body,
    torque_about_center_from_wrench,
)

Vector3 = tuple[float, float, float]


def _add(first: Vector3, second: Vector3) -> Vector3:
    return tuple(first[index] + second[index] for index in range(3))


def _subtract(first: Vector3, second: Vector3) -> Vector3:
    return tuple(first[index] - second[index] for index in range(3))


def cross(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def dot(first: Vector3, second: Vector3) -> float:
    return sum(first[index] * second[index] for index in range(3))


def norm(vector: Vector3) -> float:
    return math.sqrt(dot(vector, vector))


def normalized(vector: Vector3) -> Vector3:
    magnitude = norm(vector)
    if magnitude <= 1.0e-12:
        raise ValueError(f"cannot normalize near-zero vector; vector={vector!r}")
    return tuple(value / magnitude for value in vector)


def assert_finite_vector(vector: Vector3, *, name: str) -> None:
    if len(vector) != 3 or not all(math.isfinite(float(value)) for value in vector):
        raise AssertionError(f"{name} must be a finite Vector3; actual value={vector!r}")


def aggregate_wrenches_for_body(
    wrenches: tuple[ContactWrench, ...],
    *,
    runtime_body_id: str,
    center_world: Vector3,
) -> tuple[Vector3, Vector3]:
    aggregate = aggregate_contact_wrenches_by_body(
        tuple(wrench for wrench in wrenches if runtime_body_id in (wrench.contact.body_a, wrench.contact.body_b)),
        {
            runtime_body_id: center_world,
            **{
                other_body: center_world
                for wrench in wrenches
                for other_body in (wrench.contact.body_a, wrench.contact.body_b)
                if other_body != runtime_body_id
            },
        },
    )
    for item in aggregate:
        if item.body_id == runtime_body_id:
            return item.net_force_world, item.net_torque_world
    return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
