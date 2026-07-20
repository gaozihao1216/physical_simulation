import pytest

from physical_simulation.assets import (
    BoxGeometry,
    ColliderSpec,
    MassProperties,
    RigidBodySpec,
    Transform,
    VisualSpec,
)
from physical_simulation.validation.errors import InvalidRigidBodyError


def test_dynamic_body_without_mass_properties_raises() -> None:
    with pytest.raises(InvalidRigidBodyError, match="mass_properties"):
        RigidBodySpec(
            body_id="body",
            name="body",
            body_type="dynamic",
            transform=Transform.identity(),
            visuals=(),
            colliders=(ColliderSpec("c", BoxGeometry((1.0, 1.0, 1.0))),),
        )


def test_dynamic_body_without_collider_raises() -> None:
    with pytest.raises(InvalidRigidBodyError, match="enabled collider"):
        RigidBodySpec(
            body_id="body",
            name="body",
            body_type="dynamic",
            transform=Transform.identity(),
            visuals=(),
            colliders=(),
            mass_properties=MassProperties(1.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )


def test_static_body_can_omit_mass() -> None:
    body = RigidBodySpec(
        body_id="body",
        name="body",
        body_type="static",
        transform=Transform.identity(),
        visuals=(),
        colliders=(),
    )
    assert body.mass_properties is None


def test_duplicate_visual_id_raises() -> None:
    visual = VisualSpec("v", BoxGeometry((1.0, 1.0, 1.0)))
    with pytest.raises(InvalidRigidBodyError, match="visual IDs"):
        RigidBodySpec(
            body_id="body",
            name="body",
            body_type="static",
            transform=Transform.identity(),
            visuals=(visual, visual),
            colliders=(),
        )


def test_duplicate_collider_id_raises() -> None:
    collider = ColliderSpec("c", BoxGeometry((1.0, 1.0, 1.0)))
    with pytest.raises(InvalidRigidBodyError, match="collider IDs"):
        RigidBodySpec(
            body_id="body",
            name="body",
            body_type="static",
            transform=Transform.identity(),
            visuals=(),
            colliders=(collider, collider),
        )
