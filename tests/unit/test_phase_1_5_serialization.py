from physical_simulation.assets import create_box, create_single_body_asset
from physical_simulation.scene import AssetInstanceSpec, create_scene
from physical_simulation.serialization import (
    from_json_physics_asset,
    from_json_physics_scene,
    load_physics_asset,
    save_physics_asset,
)


def test_asset_and_scene_json_codecs(tmp_path) -> None:
    body = create_box("box_body", (1.0, 1.0, 1.0), mass=1.0)
    asset = create_single_body_asset(asset_id="box_asset", body=body)
    scene = create_scene(scene_id="scene", instances=(AssetInstanceSpec("box_01", asset),))

    assert from_json_physics_asset(asset.to_json()) == asset
    assert from_json_physics_scene(scene.to_json()) == scene

    path = tmp_path / "asset.json"
    save_physics_asset(asset, path)
    assert load_physics_asset(path) == asset
