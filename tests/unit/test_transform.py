import math

import pytest

from physical_simulation.assets import Transform
from physical_simulation.validation.errors import PhysicsValidationError


def test_identity() -> None:
    assert Transform.identity() == Transform()


def test_non_unit_quaternion_is_normalized() -> None:
    transform = Transform(rotation=(2.0, 0.0, 0.0, 0.0))
    assert transform.rotation == (1.0, 0.0, 0.0, 0.0)


def test_zero_quaternion_raises() -> None:
    with pytest.raises(PhysicsValidationError, match="rotation"):
        Transform(rotation=(0.0, 0.0, 0.0, 0.0))


def test_non_positive_scale_raises() -> None:
    with pytest.raises(PhysicsValidationError, match="scale"):
        Transform(scale=(1.0, 0.0, 1.0))


def test_nan_and_inf_raise() -> None:
    with pytest.raises(PhysicsValidationError, match="position"):
        Transform(position=(math.nan, 0.0, 0.0))
    with pytest.raises(PhysicsValidationError, match="rotation"):
        Transform(rotation=(math.inf, 0.0, 0.0, 0.0))


def test_dict_round_trip() -> None:
    transform = Transform(position=(1.0, 2.0, 3.0), rotation=(0.5, 0.5, 0.5, 0.5))
    assert Transform.from_dict(transform.to_dict()) == transform
