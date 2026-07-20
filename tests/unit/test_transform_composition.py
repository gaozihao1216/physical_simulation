import math

import pytest

from physical_simulation.assets import Transform
from physical_simulation.validation.errors import PhysicsValidationError


def _qz(degrees: float) -> tuple[float, float, float, float]:
    radians = math.radians(degrees)
    return (math.cos(radians / 2.0), 0.0, 0.0, math.sin(radians / 2.0))


def test_identity_transform_composition() -> None:
    child = Transform(position=(1.0, 2.0, 3.0), rotation=_qz(45.0))
    assert Transform.identity().compose(child) == child
    assert child.compose(Transform.identity()) == child


def test_parent_translation_is_applied() -> None:
    parent = Transform(position=(10.0, 0.0, 0.0))
    child = Transform(position=(1.0, 2.0, 3.0))
    assert parent.compose(child).position == pytest.approx((11.0, 2.0, 3.0))


def test_parent_rotation_rotates_child_position() -> None:
    half = math.sqrt(0.5)
    parent = Transform(position=(1.0, 0.0, 0.0), rotation=(half, 0.0, 0.0, half))
    child = Transform(position=(1.0, 0.0, 0.0))
    world = parent.compose(child)
    assert world.position == pytest.approx((1.0, 1.0, 0.0), abs=1e-12)


def test_parent_and_child_rotation_are_composed() -> None:
    parent = Transform(rotation=_qz(45.0))
    child = Transform(rotation=_qz(45.0))
    world = parent.compose(child)
    assert world.rotate_vector((1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)


def test_multi_level_transform_composition() -> None:
    root = Transform(position=(1.0, 0.0, 0.0), rotation=_qz(90.0))
    middle = Transform(position=(1.0, 0.0, 0.0), rotation=_qz(90.0))
    leaf = Transform(position=(1.0, 0.0, 0.0))
    world = root.compose(middle).compose(leaf)
    assert world.position == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert world.rotate_vector((1.0, 0.0, 0.0)) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)


def test_non_unit_scale_composition_fails() -> None:
    parent = Transform(scale=(2.0, 2.0, 2.0))
    child = Transform.identity()
    with pytest.raises(PhysicsValidationError, match="bake_transform_scale"):
        parent.compose(child)
    with pytest.raises(PhysicsValidationError, match="bake_transform_scale"):
        child.compose(parent)


def test_inputs_are_not_mutated_and_output_rotation_is_unit() -> None:
    parent = Transform(position=(1.0, 0.0, 0.0), rotation=_qz(30.0))
    child = Transform(position=(1.0, 0.0, 0.0), rotation=_qz(60.0))
    parent_before = parent
    child_before = child
    world = parent.compose(child)
    norm = math.sqrt(sum(component * component for component in world.rotation))
    assert parent == parent_before
    assert child == child_before
    assert norm == pytest.approx(1.0)
