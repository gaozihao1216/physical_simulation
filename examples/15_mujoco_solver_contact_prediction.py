"""Inspect MuJoCo solver contact timescale and sphere-plane collision prediction."""

from __future__ import annotations

from dataclasses import replace

from physical_simulation.assets import Transform, create_ground, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import measure_restitution
from physical_simulation.mujoco import (
    AnalyticPlane,
    MuJoCoContactSolverParams,
    estimate_solver_collision,
    estimate_solver_contact_timescale,
    predict_sphere_plane_collision,
)
from physical_simulation.scene import AssetInstanceSpec, create_scene

RUNTIME_BODY_ID = "sphere_01/sphere"
SPHERE_RADIUS = 0.1


def _body_with_params(body, params):
    return replace(body, colliders=tuple(replace(collider, mujoco_contact_params=params) for collider in body.colliders))


def _build_scene(*, timestep: float, params: MuJoCoContactSolverParams):
    ground = create_single_body_asset(
        asset_id="ground_asset",
        body=_body_with_params(create_ground("ground"), params),
    )
    sphere = create_single_body_asset(
        asset_id="sphere_asset",
        body=_body_with_params(create_sphere("sphere", SPHERE_RADIUS, mass=1.0), params),
    )
    return create_scene(
        scene_id="mujoco_solver_contact_prediction",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("sphere_01", sphere, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=timestep,
    )


def _first_prediction(params: MuJoCoContactSolverParams, *, macro_timestep: float):
    backend = MuJoCoBackend()
    backend.load_scene(_build_scene(timestep=macro_timestep, params=params))
    try:
        result = backend.reset()
        plane = AnalyticPlane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        for _ in range(240):
            state = result.get_body_state(RUNTIME_BODY_ID)
            prediction = predict_sphere_plane_collision(
                sphere_position=state.position,
                sphere_velocity=state.linear_velocity,
                sphere_radius=SPHERE_RADIUS,
                plane=plane,
                prediction_horizon=macro_timestep,
            )
            if prediction is not None:
                return result.time, state.position[2] - SPHERE_RADIUS, prediction
            result = backend.step()
    finally:
        backend.close()
    raise RuntimeError("no sphere-plane prediction found")


def _fine_contact_time(params: MuJoCoContactSolverParams) -> float:
    backend = MuJoCoBackend()
    backend.load_scene(_build_scene(timestep=1.0 / 3840.0, params=params))
    try:
        result = backend.reset()
        for _ in range(3840):
            if result.contacts:
                return result.time
            result = backend.step()
    finally:
        backend.close()
    raise RuntimeError("no fixed-fine contact found")


def main() -> None:
    macro_timestep = 1.0 / 240.0
    params = MuJoCoContactSolverParams(
        solref=(0.02, 0.5),
        solimp=(0.9, 0.9, 0.001, 0.5, 2.0),
    )
    timescale = estimate_solver_contact_timescale(params)
    macro_time, sphere_gap, prediction = _first_prediction(params, macro_timestep=macro_timestep)
    estimate = estimate_solver_collision(
        prediction=prediction,
        params=params,
        macro_timestep=macro_timestep,
    )
    fine_contact_time = _fine_contact_time(params)
    measurement_backend = MuJoCoBackend()
    measurement_backend.load_scene(_build_scene(timestep=1.0 / 3840.0, params=params))
    try:
        measurement = measure_restitution(
            measurement_backend,
            RUNTIME_BODY_ID,
            max_steps=3840,
            characteristic_length=SPHERE_RADIUS,
        )
    finally:
        measurement_backend.close()

    print(f"solref: {params.solref}")
    print(f"solimp: {params.solimp}")
    print(f"damping regime: {timescale.regime.value}")
    print(f"effective damping: {timescale.effective_damping:.6f}")
    print(f"effective stiffness: {timescale.effective_stiffness:.6f}")
    print(f"natural frequency: {timescale.natural_frequency:.6f}")
    print(f"damped frequency: {_fmt(timescale.damped_frequency)}")
    print(f"estimated contact duration: {_fmt(timescale.oscillatory_contact_duration)}")
    print(f"fastest mode timescale: {timescale.fastest_mode_timescale:.6f}")
    print(f"characteristic timescale: {timescale.characteristic_timescale:.6f}")
    print(f"sphere gap at macro sample: {sphere_gap:.6f}")
    print(f"normal approach speed: {prediction.normal_approach_speed:.6f}")
    print(f"predicted time to contact: {prediction.time_to_contact:.6f}")
    print(f"predicted absolute contact time: {macro_time + prediction.time_to_contact:.6f}")
    print(f"fixed-fine first contact time: {fine_contact_time:.6f}")
    print(f"measured contact duration: {_fmt(measurement.contact_duration_seconds)}")
    print(f"macro timestep: {macro_timestep:.9f}")
    print(f"recommended substep count: {estimate.recommendation.substep_count}")
    print(f"actual substep timestep: {estimate.recommendation.actual_substep_timestep:.9f}")
    print(f"refsafe at macro dt: {estimate.recommendation.would_trigger_refsafe_at_macro_dt}")
    print(f"timeconst satisfied at substep dt: {estimate.recommendation.satisfies_configured_timeconst_at_substep_dt}")


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
