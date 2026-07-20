import math

import pytest

from physical_simulation.math import (
    compose_pose,
    quaternion_multiply,
    quaternion_normalize,
    rotate_vector,
)
from physical_simulation.validation.errors import PhysicsValidationError


def _qz(degrees: float) -> tuple[float, float, float, float]:
    radians = math.radians(degrees)
    return (math.cos(radians / 2.0), 0.0, 0.0, math.sin(radians / 2.0))


def _qx(degrees: float) -> tuple[float, float, float, float]:
    radians = math.radians(degrees)
    return (math.cos(radians / 2.0), math.sin(radians / 2.0), 0.0, 0.0)


def test_identity_quaternion_does_not_change_vector() -> None:
    assert rotate_vector((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0)) == pytest.approx((1.0, 2.0, 3.0))


def test_z_rotation_90_degrees_rotates_x_to_y() -> None:
    assert rotate_vector(_qz(90.0), (1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)


def test_quaternion_multiply_order_is_parent_then_child() -> None:
    combined = quaternion_multiply(_qz(90.0), _qx(90.0))
    assert rotate_vector(combined, (0.0, 1.0, 0.0)) == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)


def test_compose_pose_rotates_child_position() -> None:
    position, rotation = compose_pose(
        (1.0, 0.0, 0.0),
        _qz(90.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
    )
    assert position == pytest.approx((1.0, 1.0, 0.0), abs=1e-12)
    assert rotation == pytest.approx(_qz(90.0), abs=1e-12)


def test_zero_quaternion_nan_and_inf_fail() -> None:
    with pytest.raises(PhysicsValidationError, match="quaternion"):
        quaternion_normalize((0.0, 0.0, 0.0, 0.0))
    with pytest.raises(PhysicsValidationError, match="quaternion"):
        quaternion_normalize((math.nan, 0.0, 0.0, 0.0))
    with pytest.raises(PhysicsValidationError, match="vector"):
        rotate_vector((1.0, 0.0, 0.0, 0.0), (math.inf, 0.0, 0.0))
