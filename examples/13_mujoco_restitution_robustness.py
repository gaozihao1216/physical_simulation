"""Run robust restitution measurements across solver settings and drop heights."""

from __future__ import annotations

from dataclasses import replace

from physical_simulation.assets import Transform, create_ground, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import measure_restitution_sweep
from physical_simulation.mujoco import MuJoCoContactSolverParams
from physical_simulation.scene import AssetInstanceSpec, create_scene

SPHERE_RADIUS = 0.1


def _body_with_contact_params(body, params):
    return replace(
        body,
        colliders=tuple(replace(collider, mujoco_contact_params=params) for collider in body.colliders),
    )


def _build_scene(*, solref: tuple[float, float], timestep: float, initial_height: float):
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
        body=_body_with_contact_params(create_sphere("sphere", SPHERE_RADIUS, mass=1.0), params),
    )
    return create_scene(
        scene_id="mujoco_restitution_robustness",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("sphere_01", sphere, Transform(position=(0.0, 0.0, initial_height))),
        ),
        timestep=timestep,
    )


def main() -> None:
    solrefs = ((0.02, 0.3), (0.02, 0.5), (0.02, 1.0))
    timesteps = (1.0 / 240.0, 1.0 / 480.0)
    heights = (0.4, 0.7, 1.0, 1.3)
    print(
        "solref,timestep,initial_height,impact_speed,outcome,rebound_speed,"
        "measured_restitution,maximum_penetration,normalized_penetration,"
        "duration_steps,duration_seconds"
    )
    for solref in solrefs:
        for timestep in timesteps:
            samples = measure_restitution_sweep(
                lambda height, current_solref=solref, current_timestep=timestep: _build_scene(
                    solref=current_solref,
                    timestep=current_timestep,
                    initial_height=height,
                ),
                MuJoCoBackend,
                "sphere_01/sphere",
                initial_heights=heights,
                max_steps=1400,
                characteristic_length=SPHERE_RADIUS,
            )
            for sample in samples:
                measurement = sample.measurement
                print(
                    f"{solref},{timestep:.9f},"
                    f"{sample.initial_height:.3f},"
                    f"{measurement.impact_speed:.6f},"
                    f"{measurement.outcome.value},"
                    f"{_format_optional(measurement.rebound_speed)},"
                    f"{_format_optional(measurement.measured_restitution)},"
                    f"{measurement.maximum_penetration_depth:.6f},"
                    f"{_format_optional(measurement.normalized_penetration)},"
                    f"{measurement.contact_duration_steps},"
                    f"{_format_optional(measurement.contact_duration_seconds)}"
                )


def _format_optional(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
