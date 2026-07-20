import math
from xml.etree import ElementTree as ET

import pytest

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
from physical_simulation.compilers import MUJOCO_ALL_COLLISION_BITS, MuJoCoCompiler, make_mujoco_name
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _root(scene):
    result = MuJoCoCompiler().compile(scene)
    return result, ET.fromstring(result.mjcf)


def _body_element(root, name):
    return next(body for body in root.findall(".//body") if body.attrib["name"] == name)


def _geoms(body):
    return body.findall("geom")


def test_xml_root_option_and_deterministic_compile() -> None:
    body = create_box("box_body", (0.4, 0.4, 0.4), mass=1.0)
    asset = create_single_body_asset(asset_id="box_asset", body=body)
    scene = create_scene(scene_id="box drop & test", instances=(AssetInstanceSpec("box_01", asset),))
    first = MuJoCoCompiler().compile(scene)
    second = MuJoCoCompiler().compile(scene)
    root = ET.fromstring(first.mjcf)
    assert first == second
    assert first.mjcf == second.mjcf
    assert root.tag == "mujoco"
    assert root.find("compiler").attrib["angle"] == "radian"
    assert root.find("option").attrib["gravity"] == "0 0 -9.81"
    assert root.find("option").attrib["timestep"] == "0.004166666666666667"


def test_instance_and_body_transform_are_composed_with_rotation() -> None:
    half = math.sqrt(0.5)
    body = create_box(
        "box_body",
        (0.4, 0.4, 0.4),
        mass=1.0,
        transform=Transform(position=(1.0, 0.0, 0.0), rotation=(half, 0.0, 0.0, half)),
    )
    asset = create_single_body_asset(asset_id="box_asset", body=body)
    scene = create_scene(
        scene_id="scene",
        instances=(
            AssetInstanceSpec(
                "box_01",
                asset,
                Transform(position=(1.0, 0.0, 0.0), rotation=(half, 0.0, 0.0, half)),
            ),
        ),
    )
    result, root = _root(scene)
    body_element = _body_element(root, result.get_mujoco_body_name("box_01/box_body"))
    assert body_element.attrib["pos"] == "1.0 1.0 0"
    quat = tuple(float(value) for value in body_element.attrib["quat"].split())
    assert quat == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-12)


def test_collider_and_visual_local_transforms_are_preserved() -> None:
    geometry = BoxGeometry((1.0, 1.0, 1.0))
    visual = VisualSpec("visual", geometry, Transform(position=(0.0, 1.0, 0.0)))
    collider = ColliderSpec("collider", geometry, Transform(position=(1.0, 0.0, 0.0)))
    body = RigidBodySpec(
        "body",
        "body",
        "static",
        Transform.identity(),
        (visual,),
        (collider,),
    )
    asset = create_single_body_asset(asset_id="asset", body=body)
    scene = create_scene(scene_id="scene", instances=(AssetInstanceSpec("inst", asset),))
    result, root = _root(scene)
    body_element = _body_element(root, result.get_mujoco_body_name("inst/body"))
    visual_geom = next(geom for geom in _geoms(body_element) if geom.attrib["contype"] == "0")
    collider_geom = next(geom for geom in _geoms(body_element) if geom.attrib["contype"] != "0")
    assert visual_geom.attrib["pos"] == "0 1.0 0"
    assert collider_geom.attrib["pos"] == "1.0 0 0"


def test_body_type_freejoint_strategy_and_inertial() -> None:
    dynamic = create_single_body_asset(asset_id="dynamic_asset", body=create_box("dynamic_body", (1.0, 1.0, 1.0), mass=2.0))
    fixed_dynamic = create_single_body_asset(asset_id="fixed_dynamic_asset", body=create_box("fixed_body", (1.0, 1.0, 1.0), mass=2.0))
    static = create_single_body_asset(asset_id="static_asset", body=create_ground(body_id="static_body"))
    kinematic_body = RigidBodySpec(
        "kin_body",
        "kin_body",
        "kinematic",
        Transform.identity(),
        (),
        (ColliderSpec("kin_collider", BoxGeometry((1.0, 1.0, 1.0))),),
    )
    kinematic = create_single_body_asset(asset_id="kinematic_asset", body=kinematic_body)
    scene = create_scene(
        scene_id="scene",
        instances=(
            AssetInstanceSpec("dynamic", dynamic),
            AssetInstanceSpec("fixed_dynamic", fixed_dynamic, fixed_base=True),
            AssetInstanceSpec("static", static),
            AssetInstanceSpec("kinematic", kinematic),
        ),
    )
    result, root = _root(scene)
    dynamic_el = _body_element(root, result.get_mujoco_body_name("dynamic/dynamic_body"))
    fixed_el = _body_element(root, result.get_mujoco_body_name("fixed_dynamic/fixed_body"))
    static_el = _body_element(root, result.get_mujoco_body_name("static/static_body"))
    kin_el = _body_element(root, result.get_mujoco_body_name("kinematic/kin_body"))
    assert dynamic_el.find("freejoint") is not None
    assert dynamic_el.find("inertial") is not None
    assert fixed_el.find("freejoint") is None
    assert fixed_el.find("inertial") is not None
    assert static_el.find("freejoint") is None
    assert static_el.find("inertial") is None
    assert kin_el.find("freejoint") is None
    assert kin_el.find("inertial") is None


def test_dynamic_inertial_values_and_no_geom_mass_density() -> None:
    body = create_box("box_body", (1.0, 2.0, 3.0), mass=12.0)
    asset = create_single_body_asset(asset_id="box_asset", body=body)
    scene = create_scene(scene_id="scene", instances=(AssetInstanceSpec("box", asset),))
    result, root = _root(scene)
    body_element = _body_element(root, result.get_mujoco_body_name("box/box_body"))
    inertial = body_element.find("inertial")
    assert inertial.attrib["mass"] == "12.0"
    assert inertial.attrib["pos"] == "0 0 0"
    assert inertial.attrib["diaginertia"] == "13.0 10.0 5.0"
    for geom in _geoms(body_element):
        assert "mass" not in geom.attrib
        assert "density" not in geom.attrib


def test_visual_and_collider_are_distinct_and_invisible_visual_is_skipped() -> None:
    geometry = BoxGeometry((1.0, 1.0, 1.0))
    visible = VisualSpec("visible", geometry)
    invisible = VisualSpec("invisible", geometry, visible=False)
    collider = ColliderSpec("collider", geometry)
    body = RigidBodySpec("body", "body", "static", Transform.identity(), (visible, invisible), (collider,))
    asset = create_single_body_asset(asset_id="asset", body=body)
    scene = create_scene(scene_id="scene", instances=(AssetInstanceSpec("inst", asset),))
    result, root = _root(scene)
    body_element = _body_element(root, result.get_mujoco_body_name("inst/body"))
    geoms = _geoms(body_element)
    visual_geoms = [geom for geom in geoms if geom.attrib["contype"] == "0"]
    collision_geoms = [geom for geom in geoms if geom.attrib["contype"] != "0"]
    assert len(visual_geoms) == 1
    assert visual_geoms[0].attrib["conaffinity"] == "0"
    assert len(collision_geoms) == 1
    assert visual_geoms[0].attrib["name"] != collision_geoms[0].attrib["name"]
    assert result.mujoco_geom_to_runtime_body == ((collision_geoms[0].attrib["name"], "inst/body"),)


def test_collider_material_friction_collision_mask_and_disabled_collider() -> None:
    rubber = PhysicsMaterialSpec("rubber", dynamic_friction=0.8, density=1000.0)
    geometry = BoxGeometry((1.0, 1.0, 1.0))
    enabled = ColliderSpec("enabled", geometry, material_id="rubber", collision_group=4, collision_mask=-1)
    disabled = ColliderSpec("disabled", geometry, material_id="rubber", enabled=False)
    body = RigidBodySpec("body", "body", "static", Transform.identity(), (), (enabled, disabled))
    asset = create_single_body_asset(asset_id="asset", body=body, materials=(rubber,))
    scene = create_scene(scene_id="scene", instances=(AssetInstanceSpec("inst", asset),))
    result, root = _root(scene)
    body_element = _body_element(root, result.get_mujoco_body_name("inst/body"))
    geoms = _geoms(body_element)
    assert len(geoms) == 1
    geom = geoms[0]
    assert geom.attrib["contype"] == "4"
    assert geom.attrib["conaffinity"] == str(MUJOCO_ALL_COLLISION_BITS)
    assert "-1" not in geom.attrib["conaffinity"]
    assert geom.attrib["friction"] == "0.8 0.005 0.0001"


def test_explicit_contact_pairs_use_collision_geoms_and_respect_filters() -> None:
    first_asset = create_single_body_asset(
        asset_id="first_asset",
        body=create_box("first_body", (1.0, 1.0, 1.0), body_type="static"),
    )
    blocked_body = create_box("blocked_body", (1.0, 1.0, 1.0), body_type="static")
    blocked_collider = blocked_body.colliders[0]
    blocked_body = RigidBodySpec(
        blocked_body.body_id,
        blocked_body.name,
        blocked_body.body_type,
        blocked_body.transform,
        blocked_body.visuals,
        (
            ColliderSpec(
                blocked_collider.collider_id,
                blocked_collider.geometry,
                blocked_collider.local_transform,
                blocked_collider.material_id,
                enabled=blocked_collider.enabled,
                collision_group=0,
                collision_mask=0,
            ),
        ),
        blocked_body.mass_properties,
    )
    blocked_asset = create_single_body_asset(asset_id="blocked_asset", body=blocked_body)
    scene = create_scene(
        scene_id="contact_pairs",
        instances=(
            AssetInstanceSpec("first", first_asset),
            AssetInstanceSpec("blocked", blocked_asset),
        ),
    )
    _result, root = _root(scene)

    assert root.findall(".//contact/pair") == []

    allowed_asset = create_single_body_asset(
        asset_id="allowed_asset",
        body=create_box("allowed_body", (1.0, 1.0, 1.0), body_type="static"),
    )
    scene = create_scene(
        scene_id="contact_pairs_allowed",
        instances=(
            AssetInstanceSpec("first", first_asset),
            AssetInstanceSpec("allowed", allowed_asset),
        ),
    )
    _result, root = _root(scene)
    pairs = root.findall(".//contact/pair")
    visual_names = {geom.attrib["name"] for geom in root.findall(".//geom") if geom.attrib["contype"] == "0"}

    assert len(pairs) == 1
    assert pairs[0].attrib["geom1"] not in visual_names
    assert pairs[0].attrib["geom2"] not in visual_names
    assert pairs[0].attrib["condim"] == "3"
    assert pairs[0].attrib["margin"] == "0"
    assert pairs[0].attrib["gap"] == "0"
    assert pairs[0].attrib["friction"] == "0.4 0.005 0.0001"


def test_compound_collider_has_one_inertial_and_multiple_collision_geoms() -> None:
    geometry = BoxGeometry((1.0, 1.0, 1.0))
    body = RigidBodySpec(
        "body",
        "body",
        "dynamic",
        Transform.identity(),
        (),
        (ColliderSpec("a", geometry), ColliderSpec("b", geometry, Transform(position=(1.0, 0.0, 0.0)))),
        create_box("source", (1.0, 1.0, 1.0), mass=1.0).mass_properties,
    )
    asset = create_single_body_asset(asset_id="asset", body=body)
    scene = create_scene(scene_id="scene", instances=(AssetInstanceSpec("inst", asset),))
    result, root = _root(scene)
    body_element = _body_element(root, result.get_mujoco_body_name("inst/body"))
    assert len(body_element.findall("inertial")) == 1
    assert len(_geoms(body_element)) == 2


def test_names_are_stable_safe_and_distinct_for_multiple_instances() -> None:
    body = create_box("box body 中文<&>", (1.0, 1.0, 1.0), mass=1.0)
    asset = create_single_body_asset(asset_id="asset weird", body=body)
    scene = create_scene(
        scene_id="scene<&>",
        instances=(AssetInstanceSpec("box 01", asset), AssetInstanceSpec("box 02", asset)),
    )
    first = MuJoCoCompiler().compile(scene)
    root = ET.fromstring(first.mjcf)
    names = [body.attrib["name"] for body in root.findall(".//body")]
    assert len(names) == len(set(names))
    assert make_mujoco_name("body", "box 01/box body 中文<&>") == names[0]
    assert first == MuJoCoCompiler().compile(scene)
    assert all(name.replace("_", "").isalnum() for name in names)
