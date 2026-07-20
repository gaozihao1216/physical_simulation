import math

import pytest

from physical_simulation.assets import ColliderSpec, PhysicsMaterialSpec, Transform, create_box, create_sphere
from physical_simulation.validation.errors import InvalidRigidBodyError


def test_create_box_generates_visual_and_collider() -> None:
    box = create_box("box", (1.0, 1.0, 1.0), mass=1.0)
    assert box.visuals[0].visual_id == "box_visual"
    assert box.colliders[0].collider_id == "box_collider"


def test_create_box_mass_computes_inertia() -> None:
    box = create_box("box", (1.0, 2.0, 3.0), mass=12.0)
    assert box.mass_properties is not None
    assert box.mass_properties.inertia_diagonal == pytest.approx((13.0, 10.0, 5.0))


def test_create_sphere_density_computes_mass() -> None:
    sphere = create_sphere("sphere", 1.0, density=2.0)
    assert sphere.mass_properties is not None
    assert sphere.mass_properties.mass == pytest.approx(8.0 / 3.0 * math.pi)


def test_transform_scale_must_be_baked_before_rigid_body_creation() -> None:
    with pytest.raises(InvalidRigidBodyError, match="bake_transform_scale"):
        create_box(
            "box_b",
            (1.0, 2.0, 3.0),
            mass=12.0,
            transform=Transform(scale=(10.0, 10.0, 10.0)),
        )


def test_mass_and_density_are_mutually_exclusive() -> None:
    with pytest.raises(InvalidRigidBodyError, match="mutually exclusive"):
        create_box("box", (1.0, 1.0, 1.0), mass=1.0, density=1000.0)


def test_dynamic_body_requires_mass_source() -> None:
    material = PhysicsMaterialSpec("air", density=None)
    with pytest.raises(InvalidRigidBodyError, match="requires mass"):
        create_box("box", (1.0, 1.0, 1.0), material=material)


def test_static_body_does_not_require_mass() -> None:
    box = create_box("box", (1.0, 1.0, 1.0), body_type="static")
    assert box.mass_properties is None


def test_dynamic_body_without_collider_fails_validation() -> None:
    with pytest.raises(InvalidRigidBodyError, match="enabled collider"):
        create_box("box", (1.0, 1.0, 1.0), mass=1.0, create_collider=False)


def test_dynamic_body_with_disabled_collider_fails_validation() -> None:
    collider = ColliderSpec("c", box_geometry := create_box("source", (1.0, 1.0, 1.0), mass=1.0).colliders[0].geometry, enabled=False)
    with pytest.raises(InvalidRigidBodyError, match="enabled collider"):
        from physical_simulation.assets import RigidBodySpec

        RigidBodySpec(
            body_id="body",
            name="body",
            body_type="dynamic",
            transform=Transform.identity(),
            visuals=(),
            colliders=(collider,),
            mass_properties=create_box("m", (1.0, 1.0, 1.0), mass=1.0).mass_properties,
        )
