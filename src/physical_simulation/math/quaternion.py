"""Quaternion math using project convention ``(w, x, y, z)``."""

from __future__ import annotations

import math

from physical_simulation.validation.asset_validator import _as_float_tuple
from physical_simulation.validation.errors import PhysicsValidationError


def quaternion_normalize(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Return a unit quaternion in ``(w, x, y, z)`` order."""
    q = _as_float_tuple(
        quaternion,
        field_name="quaternion",
        length=4,
        error_type=PhysicsValidationError,
    )
    norm = math.sqrt(sum(component * component for component in q))
    if norm <= 1.0e-12:
        raise PhysicsValidationError(
            f"quaternion norm must be > 1e-12; actual value={quaternion!r}"
        )
    return tuple(component / norm for component in q)


def quaternion_conjugate(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Return the conjugate of a quaternion."""
    w, x, y, z = quaternion_normalize(quaternion)
    return (w, -x, -y, -z)


def quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Return ``left * right`` using Hamilton product and normalize the result."""
    lw, lx, ly, lz = quaternion_normalize(left)
    rw, rx, ry, rz = quaternion_normalize(right)
    return quaternion_normalize(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def rotate_vector(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a 3D vector by a quaternion."""
    qw, qx, qy, qz = quaternion_normalize(quaternion)
    vx, vy, vz = _as_float_tuple(
        vector,
        field_name="vector",
        length=3,
        error_type=PhysicsValidationError,
    )

    # Optimized q * (0, v) * conjugate(q), avoiding a final normalization of the vector quaternion.
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def compose_pose(
    parent_position: tuple[float, float, float],
    parent_rotation: tuple[float, float, float, float],
    child_position: tuple[float, float, float],
    child_rotation: tuple[float, float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]:
    """Compose parent and child poses without scale.

    ``world_rotation = parent_rotation * child_rotation`` and
    ``world_position = parent_position + rotate(parent_rotation, child_position)``.
    """
    px, py, pz = _as_float_tuple(
        parent_position,
        field_name="parent_position",
        length=3,
        error_type=PhysicsValidationError,
    )
    rotated_child = rotate_vector(parent_rotation, child_position)
    world_position = (
        px + rotated_child[0],
        py + rotated_child[1],
        pz + rotated_child[2],
    )
    world_rotation = quaternion_multiply(parent_rotation, child_rotation)
    return world_position, world_rotation
