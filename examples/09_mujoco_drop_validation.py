"""Run a no-GUI MuJoCo box drop validation example."""

from __future__ import annotations

from dataclasses import asdict

from physical_simulation.assets import Transform, create_box, create_ground, create_single_body_asset
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import evaluate_resting_contact, simulate_body_trajectory
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def main() -> None:
    """Run the drop validation example."""
    ground_asset = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box_asset = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.4, 0.4, 0.4), mass=1.0),
    )
    scene = create_scene(
        scene_id="box_drop_validation_example",
        instances=(
            AssetInstanceSpec("ground_01", ground_asset, fixed_base=True),
            AssetInstanceSpec("box_01", box_asset, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=1.0 / 240.0,
    )
    runtime_id = make_runtime_body_id("box_01", "box_body")

    backend = MuJoCoBackend()
    try:
        backend.load_scene(scene)
        samples = simulate_body_trajectory(backend, runtime_id, steps=720)
        for sample in samples:
            if sample.step_index % 120 == 0:
                state = sample.state
                print(
                    f"step={sample.step_index:04d} "
                    f"time={sample.time:.3f} "
                    f"position={state.position} "
                    f"linear_velocity={state.linear_velocity} "
                    f"angular_velocity={state.angular_velocity} "
                    f"contact_count={len(sample.contacts)}"
                )

        metrics = evaluate_resting_contact(samples, runtime_id)
        print("metrics:")
        for key, value in asdict(metrics).items():
            print(f"  {key}: {value}")
        print(f"settled: {metrics.settled}")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
