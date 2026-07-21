from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_ground, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import measure_restitution
from physical_simulation.mujoco import (
    AnalyticPlane,
    MuJoCoContactSolverParams,
    estimate_solver_collision,
    predict_sphere_plane_collision,
)
from physical_simulation.scene import AssetInstanceSpec, create_scene

SPHERE_RADIUS = 0.1
RUNTIME_BODY_ID = "sphere_01/sphere"


def _body_with_params(body, params):
    return replace(body, colliders=tuple(replace(collider, mujoco_contact_params=params) for collider in body.colliders))


def _scene(*, timestep: float, params: MuJoCoContactSolverParams):
    ground = create_single_body_asset(
        asset_id="ground_asset",
        body=_body_with_params(create_ground("ground"), params),
    )
    sphere = create_single_body_asset(
        asset_id="sphere_asset",
        body=_body_with_params(create_sphere("sphere", SPHERE_RADIUS, mass=1.0, create_visual=False), params),
    )
    return create_scene(
        scene_id="solver_contact_prediction",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("sphere_01", sphere, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=timestep,
    )


def _first_macro_prediction(params: MuJoCoContactSolverParams):
    macro_timestep = 1.0 / 240.0
    backend = MuJoCoBackend()
    backend.load_scene(_scene(timestep=macro_timestep, params=params))
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
            if prediction is not None and prediction.time_to_contact <= macro_timestep:
                estimate = estimate_solver_collision(
                    prediction=prediction,
                    params=params,
                    macro_timestep=macro_timestep,
                )
                return result.time, estimate
            result = backend.step()
    finally:
        backend.close()
    raise AssertionError("expected a macro-step sphere-plane prediction before contact")


def _first_fine_contact_time(params: MuJoCoContactSolverParams) -> float:
    backend = MuJoCoBackend()
    backend.load_scene(_scene(timestep=1.0 / 3840.0, params=params))
    try:
        result = backend.reset()
        for _ in range(3840):
            if result.contacts:
                return result.time
            result = backend.step()
    finally:
        backend.close()
    raise AssertionError("expected fixed-fine sphere drop contact")


@pytest.mark.parametrize("solref", [(0.02, 0.3), (0.02, 0.5)])
def test_sphere_drop_prediction_precedes_fixed_fine_contact_and_is_deterministic(solref) -> None:
    params = MuJoCoContactSolverParams(solref=solref, solimp=(0.9, 0.9, 0.001, 0.5, 2.0))

    first_macro_time, first_estimate = _first_macro_prediction(params)
    second_macro_time, second_estimate = _first_macro_prediction(params)
    fine_contact_time = _first_fine_contact_time(params)
    predicted_absolute_time = first_macro_time + first_estimate.prediction.time_to_contact
    measurement = _measure(params)

    assert second_macro_time == pytest.approx(first_macro_time)
    assert second_estimate == first_estimate
    assert first_macro_time <= fine_contact_time
    assert predicted_absolute_time == pytest.approx(fine_contact_time, abs=1.0 / 240.0)
    assert first_estimate.prediction.normal_approach_speed > 0.0
    assert first_estimate.timescale.characteristic_timescale > 0.0
    assert first_estimate.recommendation.substep_count >= 1
    assert measurement.contact_duration_seconds is not None
    ratio = first_estimate.timescale.characteristic_timescale / measurement.contact_duration_seconds
    assert 0.25 <= ratio <= 4.0


def _measure(params: MuJoCoContactSolverParams):
    backend = MuJoCoBackend()
    backend.load_scene(_scene(timestep=1.0 / 3840.0, params=params))
    try:
        return measure_restitution(backend, RUNTIME_BODY_ID, max_steps=3840, characteristic_length=SPHERE_RADIUS)
    finally:
        backend.close()
