import pytest

from physical_simulation.assets import (
    BoxGeometry,
    ColliderSpec,
    RigidBodySpec,
    Transform,
    create_box,
    create_ground,
    create_single_body_asset,
)
from physical_simulation.compilers import MuJoCoCompiler, MuJoCoCompilationError, UnsupportedAssetStructureError
from physical_simulation.scene import AssetInstanceSpec, create_scene


def test_multi_body_asset_raises_unsupported_structure() -> None:
    ground = create_ground(body_id="ground")
    box = create_box("box", (1.0, 1.0, 1.0), mass=1.0)
    asset = create_single_body_asset(asset_id="asset", body=ground)
    object.__setattr__(asset, "bodies", (ground, box))
    scene = create_scene(scene_id="scene", instances=(AssetInstanceSpec("inst", asset),))
    with pytest.raises(UnsupportedAssetStructureError, match="exactly one rigid body"):
        MuJoCoCompiler().compile(scene)


def test_missing_material_raises_compilation_error() -> None:
    body = create_box("body", (1.0, 1.0, 1.0), mass=1.0)
    asset = create_single_body_asset(asset_id="asset", body=body)
    scene = create_scene(scene_id="scene", instances=(AssetInstanceSpec("inst", asset),))
    object.__setattr__(asset, "materials", ())
    with pytest.raises(MuJoCoCompilationError, match="material_id"):
        MuJoCoCompiler().compile(scene)


def test_collision_mask_and_group_validation() -> None:
    geometry = BoxGeometry((1.0, 1.0, 1.0))
    collider = ColliderSpec("collider", geometry)
    object.__setattr__(collider, "collision_mask", 1 << 40)
    body = RigidBodySpec("body", "body", "static", Transform.identity(), (), (collider,))
    asset = create_single_body_asset(asset_id="asset", body=body)
    scene = create_scene(scene_id="scene", instances=(AssetInstanceSpec("inst", asset),))
    with pytest.raises(MuJoCoCompilationError, match="collision_mask"):
        MuJoCoCompiler().compile(scene)

    collider = ColliderSpec("collider", geometry)
    object.__setattr__(collider, "collision_group", -2)
    body = RigidBodySpec("body", "body", "static", Transform.identity(), (), (collider,))
    asset = create_single_body_asset(asset_id="asset", body=body)
    scene = create_scene(scene_id="scene", instances=(AssetInstanceSpec("inst", asset),))
    with pytest.raises(MuJoCoCompilationError, match="collision_group"):
        MuJoCoCompiler().compile(scene)
