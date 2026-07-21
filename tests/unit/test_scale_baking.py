import math

import pytest

from physical_simulation.assets import (
    BoxGeometry,
    CapsuleGeometry,
    ColliderSpec,
    ConeGeometry,
    CylinderGeometry,
    EllipsoidGeometry,
    FrustumGeometry,
    RegularPrismGeometry,
    SphericalCapGeometry,
    RigidBodySpec,
    SphereGeometry,
    Transform,
    VisualSpec,
    WedgeGeometry,
    bake_scale_into_geometry,
    bake_transform_scale,
    create_box,
)
from physical_simulation.validation.errors import InvalidRigidBodyError, PhysicsValidationError, ScaleBakingError


def test_unit_scale_returns_equivalent_geometry() -> None:
    geometry = BoxGeometry((1.0, 2.0, 3.0))
    assert bake_scale_into_geometry(geometry, (1.0, 1.0, 1.0)) == geometry


def test_box_non_uniform_scale() -> None:
    assert bake_scale_into_geometry(BoxGeometry((1.0, 2.0, 3.0)), (2.0, 3.0, 4.0)) == BoxGeometry(
        (2.0, 6.0, 12.0)
    )
    assert bake_scale_into_geometry(WedgeGeometry((1.0, 2.0, 3.0)), (2.0, 3.0, 4.0)) == WedgeGeometry(
        (2.0, 6.0, 12.0)
    )
    assert bake_scale_into_geometry(EllipsoidGeometry((1.0, 2.0, 3.0)), (2.0, 3.0, 4.0)) == EllipsoidGeometry(
        (2.0, 6.0, 12.0)
    )


def test_sphere_uniform_scale_and_non_uniform_error() -> None:
    assert bake_scale_into_geometry(SphereGeometry(0.5), (2.0, 2.0, 2.0)) == SphereGeometry(1.0)
    with pytest.raises(ScaleBakingError, match="ellipsoid"):
        bake_scale_into_geometry(SphereGeometry(0.5), (2.0, 1.0, 2.0))


def test_cylinder_radial_scale_rules() -> None:
    assert bake_scale_into_geometry(CylinderGeometry(0.5, 2.0), (2.0, 2.0, 3.0)) == CylinderGeometry(1.0, 6.0)
    with pytest.raises(ScaleBakingError, match="elliptical cylinder"):
        bake_scale_into_geometry(CylinderGeometry(0.5, 2.0), (2.0, 3.0, 1.0))
    assert bake_scale_into_geometry(ConeGeometry(0.5, 2.0), (2.0, 2.0, 3.0)) == ConeGeometry(1.0, 6.0)
    with pytest.raises(ScaleBakingError, match="elliptical cone"):
        bake_scale_into_geometry(ConeGeometry(0.5, 2.0), (2.0, 3.0, 1.0))
    assert bake_scale_into_geometry(FrustumGeometry(0.5, 0.25, 2.0), (2.0, 2.0, 3.0)) == FrustumGeometry(
        1.0, 0.5, 6.0
    )
    with pytest.raises(ScaleBakingError, match="elliptical frustum"):
        bake_scale_into_geometry(FrustumGeometry(0.5, 0.25, 2.0), (2.0, 3.0, 1.0))
    assert bake_scale_into_geometry(RegularPrismGeometry(6, 0.5, 2.0), (2.0, 2.0, 3.0)) == RegularPrismGeometry(
        6, 1.0, 6.0
    )
    with pytest.raises(ScaleBakingError, match="regular polygon"):
        bake_scale_into_geometry(RegularPrismGeometry(6, 0.5, 2.0), (2.0, 3.0, 1.0))


def test_capsule_requires_uniform_scale() -> None:
    assert bake_scale_into_geometry(CapsuleGeometry(0.5, 2.0), (2.0, 2.0, 2.0)) == CapsuleGeometry(1.0, 4.0)
    with pytest.raises(ScaleBakingError, match="uniform"):
        bake_scale_into_geometry(CapsuleGeometry(0.5, 2.0), (2.0, 2.0, 3.0))
    assert bake_scale_into_geometry(SphericalCapGeometry(1.0, 0.25), (2.0, 2.0, 2.0)) == SphericalCapGeometry(
        2.0, 0.5
    )
    with pytest.raises(ScaleBakingError, match="ellipsoidal cap"):
        bake_scale_into_geometry(SphericalCapGeometry(1.0, 0.25), (2.0, 2.0, 3.0))


def test_bake_transform_scale_keeps_pose_and_does_not_mutate_inputs() -> None:
    geometry = BoxGeometry((1.0, 2.0, 3.0))
    transform = Transform(position=(1.0, 2.0, 3.0), rotation=(0.5, 0.5, 0.5, 0.5), scale=(2.0, 3.0, 4.0))
    baked_geometry, baked_transform = bake_transform_scale(geometry, transform)
    assert baked_geometry == BoxGeometry((2.0, 6.0, 12.0))
    assert baked_transform.position == transform.position
    assert baked_transform.rotation == transform.rotation
    assert baked_transform.scale == (1.0, 1.0, 1.0)
    assert geometry == BoxGeometry((1.0, 2.0, 3.0))
    assert transform.scale == (2.0, 3.0, 4.0)


def test_non_unit_body_and_collider_scale_fail_but_visual_scale_is_allowed() -> None:
    with pytest.raises(InvalidRigidBodyError, match="bake_transform_scale"):
        RigidBodySpec(
            body_id="b",
            name="b",
            body_type="static",
            transform=Transform(scale=(2.0, 2.0, 2.0)),
            visuals=(),
            colliders=(),
        )
    with pytest.raises(PhysicsValidationError, match="ColliderSpec"):
        ColliderSpec("c", BoxGeometry((1.0, 1.0, 1.0)), local_transform=Transform(scale=(2.0, 2.0, 2.0)))
    visual = VisualSpec("v", BoxGeometry((1.0, 1.0, 1.0)), local_transform=Transform(scale=(2.0, 3.0, 4.0)))
    assert visual.local_transform.scale == (2.0, 3.0, 4.0)


def test_invalid_scale_values_raise() -> None:
    with pytest.raises(ScaleBakingError, match="scale"):
        bake_scale_into_geometry(BoxGeometry((1.0, 1.0, 1.0)), (math.inf, 1.0, 1.0))
    with pytest.raises(PhysicsValidationError, match="scale"):
        Transform(scale=(1.0, -1.0, 1.0))
