"""Test-only contact wrench aggregation helpers."""

from __future__ import annotations

import math

from physical_simulation.runtime import ContactWrench

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


def force_on_body(wrench: ContactWrench, runtime_body_id: str) -> Vector3:
    if wrench.contact.body_a == runtime_body_id:
        return wrench.force_on_body_a_world
    if wrench.contact.body_b == runtime_body_id:
        return wrench.force_on_body_b_world
    raise ValueError(
        f"wrench does not involve runtime body; runtime_body_id={runtime_body_id!r}, "
        f"body_a={wrench.contact.body_a!r}, body_b={wrench.contact.body_b!r}"
    )


def pure_contact_torque_on_body(wrench: ContactWrench, runtime_body_id: str) -> Vector3:
    if wrench.contact.body_a == runtime_body_id:
        return wrench.contact_torque_on_body_a_world
    if wrench.contact.body_b == runtime_body_id:
        return wrench.contact_torque_on_body_b_world
    raise ValueError(
        f"wrench does not involve runtime body; runtime_body_id={runtime_body_id!r}, "
        f"body_a={wrench.contact.body_a!r}, body_b={wrench.contact.body_b!r}"
    )


def torque_about_center_from_wrench(
    wrench: ContactWrench,
    *,
    runtime_body_id: str,
    center_world: Vector3,
) -> Vector3:
    force = force_on_body(wrench, runtime_body_id)
    pure_torque = pure_contact_torque_on_body(wrench, runtime_body_id)
    lever = _subtract(wrench.contact.position, center_world)
    return _add(pure_torque, cross(lever, force))


def aggregate_wrenches_for_body(
    wrenches: tuple[ContactWrench, ...],
    *,
    runtime_body_id: str,
    center_world: Vector3,
) -> tuple[Vector3, Vector3]:
    net_force = (0.0, 0.0, 0.0)
    net_torque = (0.0, 0.0, 0.0)
    for wrench in wrenches:
        if runtime_body_id not in (wrench.contact.body_a, wrench.contact.body_b):
            continue
        net_force = _add(net_force, force_on_body(wrench, runtime_body_id))
        net_torque = _add(
            net_torque,
            torque_about_center_from_wrench(
                wrench,
                runtime_body_id=runtime_body_id,
                center_world=center_world,
            ),
        )
    return net_force, net_torque
