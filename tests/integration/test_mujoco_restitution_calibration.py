from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_ground, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import ReferenceRestitutionTarget, measure_restitution
from physical_simulation.mujoco import MuJoCoContactSolverParams
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _body_with_params(body, params):
    return replace(body, colliders=tuple(replace(collider, mujoco_contact_params=params) for collider in body.colliders))


def _drop_measurement(solref, *, timestep=1.0 / 240.0):
    params = MuJoCoContactSolverParams(solref=solref, solimp=(0.9, 0.95, 0.001, 0.5, 2.0))
    ground = create_single_body_asset(
        asset_id="ground_asset",
        body=_body_with_params(create_ground("ground"), params),
    )
    sphere = create_single_body_asset(
        asset_id="sphere_asset",
        body=_body_with_params(create_sphere("sphere", 0.1, mass=1.0), params),
    )
    scene = create_scene(
        scene_id="restitution_calibration",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("sphere_01", sphere, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=timestep,
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    try:
        return measure_restitution(backend, "sphere_01/sphere", max_steps=1000)
    finally:
        backend.close()


def test_reference_restitution_target_is_not_solver_params() -> None:
    target = ReferenceRestitutionTarget(restitution=0.5, reference_impact_speed=2.0)

    assert target.restitution == 0.5
    assert target.reference_impact_speed == 2.0
    with pytest.raises(ValueError, match="restitution"):
        ReferenceRestitutionTarget(restitution=1.5, reference_impact_speed=2.0)


def test_measure_restitution_reports_non_negative_response() -> None:
    measurement = _drop_measurement((0.02, 0.5))

    assert measurement.impact_speed > 0.0
    assert measurement.rebound_speed >= 0.0
    assert measurement.measured_restitution >= 0.0
    assert measurement.contact_start_step is not None
    assert measurement.contact_end_step is not None
    assert measurement.maximum_penetration_depth > 0.0
    assert measurement.contact_duration_steps > 0


def test_lower_damping_ratio_produces_more_rebound_than_critical_damping() -> None:
    underdamped = _drop_measurement((0.02, 0.3))
    critically_damped = _drop_measurement((0.02, 1.0))

    assert underdamped.rebound_speed > critically_damped.rebound_speed
    assert underdamped.measured_restitution > critically_damped.measured_restitution


def test_timestep_can_change_measured_restitution() -> None:
    coarse = _drop_measurement((0.02, 0.5), timestep=1.0 / 240.0)
    fine = _drop_measurement((0.02, 0.5), timestep=1.0 / 480.0)

    assert coarse.impact_speed == pytest.approx(fine.impact_speed, rel=0.02)
    assert abs(coarse.measured_restitution - fine.measured_restitution) > 1.0e-4
