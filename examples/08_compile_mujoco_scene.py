"""Compile a simple box-drop scene to MJCF without running MuJoCo."""

from pathlib import Path

from physical_simulation.assets import Transform, create_box, create_ground, create_single_body_asset
from physical_simulation.compilers import MuJoCoCompiler
from physical_simulation.scene import AssetInstanceSpec, create_scene


def main() -> None:
    """Run the MJCF compiler example."""
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
        scene_id="box_drop",
        instances=(
            AssetInstanceSpec("ground_01", ground_asset, fixed_base=True),
            AssetInstanceSpec("box_01", box_asset, Transform(position=(0.0, 0.0, 1.0))),
        ),
    )

    result = MuJoCoCompiler().compile(scene)
    print("Runtime body to MuJoCo body name:")
    for runtime_body_id, mujoco_name in result.runtime_body_to_mujoco_name:
        print(f"  {runtime_body_id} -> {mujoco_name}")
    print(result.mjcf)

    output_path = Path("outputs/examples/box_drop.xml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.mjcf, encoding="utf-8")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
