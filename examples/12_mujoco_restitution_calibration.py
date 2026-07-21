"""Scan MuJoCo solref values in a simple sphere-drop restitution experiment."""

from __future__ import annotations

from dataclasses import replace

from physical_simulation.assets import Transform, create_ground, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import measure_restitution
from physical_simulation.mujoco import MuJoCoContactSolverParams
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _body_with_contact_params(body, params):
    return replace(
        body,
        colliders=tuple(replace(collider, mujoco_contact_params=params) for collider in body.colliders),
    )


def _build_scene(*, solref: tuple[float, float], timestep: float):
    params = MuJoCoContactSolverParams(
        solref=solref,
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
    )
    ground = create_single_body_asset(
        asset_id="ground_asset",
        body=_body_with_contact_params(create_ground("ground"), params),
    )
    sphere = create_single_body_asset(
        asset_id="sphere_asset",
        body=_body_with_contact_params(create_sphere("sphere", 0.1, mass=1.0), params),
    )
    return create_scene(
        scene_id="mujoco_restitution_calibration",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("sphere_01", sphere, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=timestep,
    )


def main() -> None:
    solrefs = ((0.02, 0.3), (0.02, 0.5), (0.02, 1.0))
    timesteps = (1.0 / 240.0, 1.0 / 480.0)
    print(
        "solref,timestep,impact_speed,rebound_speed,measured_restitution,"
        "maximum_penetration,contact_duration_steps"
    )
    for solref in solrefs:
        for timestep in timesteps:
            backend = MuJoCoBackend()
            backend.load_scene(_build_scene(solref=solref, timestep=timestep))
            try:
                measurement = measure_restitution(backend, "sphere_01/sphere", max_steps=1000)
            finally:
                backend.close()
            print(
                f"{solref},{timestep:.9f},"
                f"{measurement.impact_speed:.6f},"
                f"{measurement.rebound_speed:.6f},"
                f"{measurement.measured_restitution:.6f},"
                f"{measurement.maximum_penetration_depth:.6f},"
                f"{measurement.contact_duration_steps}"
            )


if __name__ == "__main__":
    main()
