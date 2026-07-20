from __future__ import annotations

import math
from xml.etree import ElementTree as ET

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import (
    BoxGeometry,
    ColliderSpec,
    PhysicsMaterialSpec,
    RigidBodySpec,
    Transform,
    VisualSpec,
    create_box,
    create_ground,
    create_single_body_asset,
)
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.compilers import (
    MUJOCO_EXPLICIT_PAIR_CONDIM,
    MUJOCO_EXPLICIT_PAIR_GAP,
    MUJOCO_EXPLICIT_PAIR_MARGIN,
    MUJOCO_ROLLING_FRICTION,
    MUJOCO_TORSIONAL_FRICTION,
    MuJoCoCompiler,
)
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _pairs(scene):
    result = MuJoCoCompiler().compile(scene)
    return result, ET.fromstring(result.mjcf).findall(".//contact/pair")


def _static_box_asset(asset_id: str, body_id: str, material: PhysicsMaterialSpec | None = None):
    body = create_box(
        body_id,
        (0.2, 0.2, 0.2),
        body_type="static",
        material=material,
    )
    return create_single_body_asset(
        asset_id=asset_id,
        body=body,
        materials=None if material is None else (material,),
    )


def test_fixed_fixed_pair_has_explicit_contact_parameters_and_loads() -> None:
    rubber = PhysicsMaterialSpec("rubber", dynamic_friction=0.2)
    steel = PhysicsMaterialSpec("steel", dynamic_friction=0.8)
    first = _static_box_asset("first_asset", "first_body", rubber)
    second = create_single_body_asset(
        asset_id="second_asset",
        body=create_box("second_body", (0.2, 0.2, 0.2), mass=1.0, material=steel),
        materials=(steel,),
    )
    scene = create_scene(
        scene_id="fixed_fixed_pair",
        instances=(
            AssetInstanceSpec("first", first),
            AssetInstanceSpec("second", second, Transform(position=(0.0, 0.0, 0.15)), fixed_base=True),
        ),
    )

    result, pairs = _pairs(scene)
    assert result == MuJoCoCompiler().compile(scene)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.attrib["geom1"] < pair.attrib["geom2"]
    assert pair.attrib["condim"] == str(MUJOCO_EXPLICIT_PAIR_CONDIM)
    assert pair.attrib["margin"] == "0"
    assert pair.attrib["gap"] == "0"
    friction = tuple(float(value) for value in pair.attrib["friction"].split())
    assert friction == pytest.approx(
        (
            math.sqrt(rubber.dynamic_friction * steel.dynamic_friction),
            MUJOCO_TORSIONAL_FRICTION,
            MUJOCO_ROLLING_FRICTION,
        )
    )

    backend = MuJoCoBackend()
    backend.load_scene(scene)
    contacts = backend.reset().contacts
    assert contacts
    assert all({contact.body_a, contact.body_b} == {"first/first_body", "second/second_body"} for contact in contacts)


def test_dynamic_static_uses_automatic_collision_without_explicit_pair() -> None:
    ground = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.2, 0.2, 0.2), mass=1.0),
    )
    scene = create_scene(
        scene_id="dynamic_static_auto",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("box_01", box, Transform(position=(0.0, 0.0, 0.09))),
        ),
    )

    _result, pairs = _pairs(scene)
    assert pairs == []
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    assert backend.reset().contacts


def test_dynamic_dynamic_uses_automatic_collision_without_explicit_pair() -> None:
    asset = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.2, 0.2, 0.2), mass=1.0),
    )
    scene = create_scene(
        scene_id="dynamic_dynamic_auto",
        instances=(
            AssetInstanceSpec("box_01", asset, Transform(position=(0.0, 0.0, 0.0))),
            AssetInstanceSpec("box_02", asset, Transform(position=(0.0, 0.0, 0.15))),
        ),
    )

    _result, pairs = _pairs(scene)
    assert pairs == []
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    assert backend.reset().contacts


def test_fixed_fixed_mask_blocks_pair_and_contact() -> None:
    allowed = _static_box_asset("allowed_asset", "allowed_body")
    blocked_source = create_box("blocked_body", (0.2, 0.2, 0.2), body_type="static")
    collider = blocked_source.colliders[0]
    blocked_body = RigidBodySpec(
        blocked_source.body_id,
        blocked_source.name,
        blocked_source.body_type,
        blocked_source.transform,
        blocked_source.visuals,
        (
            ColliderSpec(
                collider.collider_id,
                collider.geometry,
                collider.local_transform,
                collider.material_id,
                collision_group=0,
                collision_mask=0,
            ),
        ),
        blocked_source.mass_properties,
    )
    blocked = create_single_body_asset(asset_id="blocked_asset", body=blocked_body)
    scene = create_scene(
        scene_id="fixed_fixed_mask_blocked",
        instances=(
            AssetInstanceSpec("allowed", allowed),
            AssetInstanceSpec("blocked", blocked, Transform(position=(0.0, 0.0, 0.15))),
        ),
    )

    _result, pairs = _pairs(scene)
    assert pairs == []
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    assert backend.reset().contacts == ()


def test_visual_geoms_and_same_body_colliders_do_not_generate_pairs() -> None:
    geometry = BoxGeometry((0.2, 0.2, 0.2))
    body = RigidBodySpec(
        "compound_body",
        "compound_body",
        "static",
        Transform.identity(),
        (VisualSpec("visual", geometry),),
        (
            ColliderSpec("left", geometry, Transform(position=(-0.05, 0.0, 0.0))),
            ColliderSpec("right", geometry, Transform(position=(0.05, 0.0, 0.0))),
        ),
    )
    compound = create_single_body_asset(asset_id="compound_asset", body=body)
    other = _static_box_asset("other_asset", "other_body")
    scene = create_scene(
        scene_id="visual_and_same_body_pairs",
        instances=(
            AssetInstanceSpec("compound", compound),
            AssetInstanceSpec("other", other, Transform(position=(0.0, 0.0, 0.15))),
        ),
    )

    _result, pairs = _pairs(scene)
    visual_names = {
        geom.attrib["name"]
        for geom in ET.fromstring(MuJoCoCompiler().compile(scene).mjcf).findall(".//geom")
        if geom.attrib["contype"] == "0"
    }
    assert len(pairs) == 2
    assert all(pair.attrib["geom1"] not in visual_names for pair in pairs)
    assert all(pair.attrib["geom2"] not in visual_names for pair in pairs)


def test_material_mixing_is_symmetric_when_scene_order_changes() -> None:
    low = PhysicsMaterialSpec("low", dynamic_friction=0.25)
    high = PhysicsMaterialSpec("high", dynamic_friction=1.0)
    low_asset = _static_box_asset("low_asset", "low_body", low)
    high_asset = _static_box_asset("high_asset", "high_body", high)
    first_scene = create_scene(
        scene_id="material_symmetry_a",
        instances=(
            AssetInstanceSpec("low", low_asset),
            AssetInstanceSpec("high", high_asset, Transform(position=(0.0, 0.0, 0.15))),
        ),
    )
    second_scene = create_scene(
        scene_id="material_symmetry_b",
        instances=(
            AssetInstanceSpec("high", high_asset, Transform(position=(0.0, 0.0, 0.15))),
            AssetInstanceSpec("low", low_asset),
        ),
    )

    _first, first_pairs = _pairs(first_scene)
    _second, second_pairs = _pairs(second_scene)
    assert len(first_pairs) == len(second_pairs) == 1
    assert first_pairs[0].attrib["friction"] == second_pairs[0].attrib["friction"]
    assert tuple(float(value) for value in first_pairs[0].attrib["friction"].split())[0] == pytest.approx(0.5)


def test_pair_count_only_tracks_fixed_fixed_combinations() -> None:
    fixed_a = _static_box_asset("fixed_a_asset", "fixed_a_body")
    fixed_b = _static_box_asset("fixed_b_asset", "fixed_b_body")
    dynamic = create_single_body_asset(
        asset_id="dynamic_asset",
        body=create_box("dynamic_body", (0.2, 0.2, 0.2), mass=1.0),
    )
    scene = create_scene(
        scene_id="pair_count",
        instances=(
            AssetInstanceSpec("fixed_a", fixed_a),
            AssetInstanceSpec("fixed_b", fixed_b, Transform(position=(0.0, 0.0, 0.15))),
            AssetInstanceSpec("dynamic_01", dynamic, Transform(position=(1.0, 0.0, 0.0))),
            AssetInstanceSpec("dynamic_02", dynamic, Transform(position=(2.0, 0.0, 0.0))),
            AssetInstanceSpec("dynamic_03", dynamic, Transform(position=(3.0, 0.0, 0.0))),
        ),
    )

    _result, pairs = _pairs(scene)
    assert len(pairs) == 1
