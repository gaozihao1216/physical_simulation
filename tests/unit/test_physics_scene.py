import math

import pytest

from physical_simulation.assets import Transform, create_box, create_ground, create_single_body_asset
from physical_simulation.scene import AssetInstanceSpec, PhysicsSceneSpec, create_scene
from physical_simulation.serialization import load_physics_scene, save_physics_scene
from physical_simulation.validation.errors import InvalidPhysicsSceneError


def _instance(instance_id: str = "box_01") -> AssetInstanceSpec:
    asset = create_single_body_asset(asset_id="box_asset", body=create_box("box_body", (1.0, 1.0, 1.0), mass=1.0))
    return AssetInstanceSpec(instance_id, asset)


def test_single_and_multiple_instance_scene_creation() -> None:
    first = _instance("box_01")
    second = AssetInstanceSpec("box_02", first.asset, Transform(position=(1.0, 0.0, 0.0)))
    scene = create_scene(scene_id="scene", instances=(first, second))
    assert len(scene.instances) == 2
    assert PhysicsSceneSpec.from_dict(scene.to_dict()) == scene
    assert PhysicsSceneSpec.from_json(scene.to_json()) == scene


def test_scene_validation_errors() -> None:
    instance = _instance()
    with pytest.raises(InvalidPhysicsSceneError, match="scene_id"):
        create_scene(scene_id="", instances=(instance,))
    with pytest.raises(InvalidPhysicsSceneError, match="schema_version"):
        PhysicsSceneSpec("2.0", "scene", (0.0, 0.0, -9.81), 1 / 240, (instance,))
    with pytest.raises(InvalidPhysicsSceneError, match="gravity"):
        PhysicsSceneSpec("1.0", "scene", (0.0, 0.0), 1 / 240, (instance,))
    with pytest.raises(InvalidPhysicsSceneError, match="gravity"):
        PhysicsSceneSpec("1.0", "scene", (0.0, math.inf, 0.0), 1 / 240, (instance,))
    for timestep in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(InvalidPhysicsSceneError, match="timestep"):
            PhysicsSceneSpec("1.0", "scene", (0.0, 0.0, -9.81), timestep, (instance,))
    with pytest.raises(InvalidPhysicsSceneError, match="instances"):
        create_scene(scene_id="scene", instances=())
    with pytest.raises(InvalidPhysicsSceneError, match="instance_id"):
        create_scene(scene_id="scene", instances=(instance, instance))
    with pytest.raises(InvalidPhysicsSceneError, match="metadata"):
        create_scene(scene_id="scene", instances=(instance,), metadata={"ok": 1})  # type: ignore[arg-type]


def test_scene_file_round_trip_and_reused_asset(tmp_path) -> None:
    ground_asset = create_single_body_asset(asset_id="ground_asset", body=create_ground(body_id="ground_body"))
    box_asset = create_single_body_asset(asset_id="box_asset", body=create_box("box_body", (1.0, 1.0, 1.0), mass=1.0))
    scene = create_scene(
        scene_id="scene",
        instances=(
            AssetInstanceSpec("ground_01", ground_asset),
            AssetInstanceSpec("box_01", box_asset, Transform(position=(-0.5, 0.0, 1.0))),
            AssetInstanceSpec("box_02", box_asset, Transform(position=(0.5, 0.0, 2.0))),
        ),
    )
    path = tmp_path / "scene.json"
    save_physics_scene(scene, path)
    assert load_physics_scene(path) == scene
