"""Demonstrate placing reusable assets into a PhysicsSceneSpec."""

from physical_simulation.assets import Transform, create_box, create_ground, create_single_body_asset
from physical_simulation.scene import AssetInstanceSpec, PhysicsSceneSpec, create_scene


def main() -> None:
    """Run the scene instance example."""
    ground_asset = create_single_body_asset(
        asset_id="ground_asset",
        body=create_ground(body_id="ground_body"),
        name="Ground",
    )
    box_asset = create_single_body_asset(
        asset_id="box_asset",
        body=create_box(body_id="box_body", size=(0.4, 0.4, 0.4), mass=1.0),
        name="Box",
    )
    scene = create_scene(
        scene_id="two_boxes",
        instances=(
            AssetInstanceSpec("ground_01", ground_asset),
            AssetInstanceSpec("box_01", box_asset, Transform(position=(-0.5, 0.0, 1.0))),
            AssetInstanceSpec("box_02", box_asset, Transform(position=(0.5, 0.0, 2.0))),
        ),
    )
    print(scene.to_json())
    restored = PhysicsSceneSpec.from_json(scene.to_json())
    print(f"round_trip_equal: {restored == scene}")


if __name__ == "__main__":
    main()
