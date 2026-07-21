from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest

from physical_simulation.assets import BoxGeometry, ColliderSpec, RigidBodySpec, Transform, create_box, create_single_body_asset
from physical_simulation.compilers import MuJoCoCompiler
from physical_simulation.mujoco import MuJoCoContactSolverParams
from physical_simulation.scene import AssetInstanceSpec, create_scene
from physical_simulation.validation.errors import PhysicsValidationError


def _params(
    *,
    solref=(0.02, 0.5),
    solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
    margin=0.001,
    gap=0.0,
    priority=0,
    solmix=1.0,
) -> MuJoCoContactSolverParams:
    return MuJoCoContactSolverParams(
        solref=solref,
        solimp=solimp,
        margin=margin,
        gap=gap,
        priority=priority,
        solmix=solmix,
    )


def _root(scene):
    return ET.fromstring(MuJoCoCompiler().compile(scene).mjcf)


def _collision_geoms(root):
    return tuple(geom for geom in root.findall(".//geom") if geom.attrib.get("contype") != "0")


def test_mujoco_contact_solver_params_validation_and_round_trip() -> None:
    params = _params(priority=2, solmix=0.25)

    assert MuJoCoContactSolverParams.from_dict(params.to_dict()) == params
    with pytest.raises(PhysicsValidationError, match="solref"):
        _params(solref=(0.02,))
    with pytest.raises(PhysicsValidationError, match="margin"):
        _params(margin=-0.1)
    with pytest.raises(PhysicsValidationError, match="priority"):
        _params(priority=-1)


def test_collider_serializes_optional_mujoco_contact_params() -> None:
    collider = ColliderSpec("collider", BoxGeometry((1.0, 1.0, 1.0)), mujoco_contact_params=_params())

    assert ColliderSpec.from_dict(collider.to_dict()) == collider
    with pytest.raises(PhysicsValidationError, match="mujoco_contact_params"):
        ColliderSpec("bad", BoxGeometry((1.0, 1.0, 1.0)), mujoco_contact_params=object())  # type: ignore[arg-type]


def test_dynamic_geom_writes_mujoco_contact_params() -> None:
    params = _params(solref=(0.02, 0.3), margin=0.002, gap=0.001, priority=3, solmix=0.5)
    body = RigidBodySpec(
        "body",
        "body",
        "dynamic",
        Transform.identity(),
        (),
        (ColliderSpec("collider", BoxGeometry((1.0, 1.0, 1.0)), mujoco_contact_params=params),),
        create_box("mass_source", (1.0, 1.0, 1.0), mass=1.0).mass_properties,
    )
    scene = create_scene(
        scene_id="solver_params_geom",
        instances=(AssetInstanceSpec("inst", create_single_body_asset(asset_id="asset", body=body)),),
    )

    geom = _collision_geoms(_root(scene))[0]

    assert geom.attrib["solref"] == "0.02 0.3"
    assert geom.attrib["solimp"] == "0.9 0.95 0.001 0.5 2.0"
    assert geom.attrib["margin"] == "0.002"
    assert geom.attrib["gap"] == "0.001"
    assert geom.attrib["priority"] == "3"
    assert geom.attrib["solmix"] == "0.5"


def test_unconfigured_geom_keeps_existing_defaults() -> None:
    scene = create_scene(
        scene_id="default_solver_params",
        instances=(AssetInstanceSpec("inst", create_single_body_asset(asset_id="asset", body=create_box("box", (1.0, 1.0, 1.0), mass=1.0))),),
    )

    geom = _collision_geoms(_root(scene))[0]

    for name in ("solref", "solimp", "margin", "gap", "priority", "solmix"):
        assert name not in geom.attrib


def test_explicit_pair_resolves_solver_params_because_pair_overrides_geom() -> None:
    low_priority = _params(solref=(0.02, 1.0), solimp=(0.8, 0.9, 0.001, 0.5, 2.0), margin=0.001)
    high_priority = _params(
        solref=(0.02, 0.3),
        solimp=(0.7, 0.95, 0.002, 0.5, 2.0),
        margin=0.002,
        gap=0.001,
        priority=2,
    )
    first_body = RigidBodySpec(
        "first_body",
        "first_body",
        "static",
        Transform.identity(),
        (),
        (ColliderSpec("collider", BoxGeometry((0.2, 0.2, 0.2)), mujoco_contact_params=low_priority),),
    )
    second_body = RigidBodySpec(
        "second_body",
        "second_body",
        "static",
        Transform.identity(),
        (),
        (ColliderSpec("collider", BoxGeometry((0.2, 0.2, 0.2)), mujoco_contact_params=high_priority),),
    )
    scene = create_scene(
        scene_id="explicit_pair_solver_params",
        instances=(
            AssetInstanceSpec("first", create_single_body_asset(asset_id="first_asset", body=first_body)),
            AssetInstanceSpec("second", create_single_body_asset(asset_id="second_asset", body=second_body), Transform(position=(0.0, 0.0, 0.15))),
        ),
    )

    pair = _root(scene).find(".//contact/pair")

    assert pair is not None
    assert pair.attrib["solref"] == "0.02 0.3"
    assert pair.attrib["solimp"] == "0.7 0.95 0.002 0.5 2.0"
    assert pair.attrib["margin"] == "0.003"
    assert pair.attrib["gap"] == "0.001"
