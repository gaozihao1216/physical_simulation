from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_box, create_ground, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import RestitutionOutcome, measure_restitution
from physical_simulation.mujoco import MuJoCoContactSolverParams, MuJoCoSubstepRunner
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _load_backend(scene) -> MuJoCoBackend:
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    return backend


def _freefall_scene(*, timestep: float):
    box = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box", (0.2, 0.2, 0.2), mass=1.0, create_visual=False),
    )
    return create_scene(
        scene_id="substep_freefall",
        instances=(AssetInstanceSpec("box_01", box, Transform(position=(0.0, 0.0, 2.0))),),
        timestep=timestep,
    )


def _body_with_params(body, params):
    return replace(body, colliders=tuple(replace(collider, mujoco_contact_params=params) for collider in body.colliders))


def _sphere_drop_scene(*, timestep: float, solref=(0.02, 0.5), initial_height=1.0):
    params = MuJoCoContactSolverParams(solref=solref, solimp=(0.9, 0.95, 0.001, 0.5, 2.0))
    ground = create_single_body_asset(
        asset_id="ground_asset",
        body=_body_with_params(create_ground("ground"), params),
    )
    sphere = create_single_body_asset(
        asset_id="sphere_asset",
        body=_body_with_params(create_sphere("sphere", 0.1, mass=1.0, create_visual=False), params),
    )
    return create_scene(
        scene_id="substep_sphere_drop",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("sphere_01", sphere, Transform(position=(0.0, 0.0, initial_height))),
        ),
        timestep=timestep,
    )


def test_substepped_freefall_matches_fixed_fine_timestep() -> None:
    runtime_id = "box_01/box"
    fine = _load_backend(_freefall_scene(timestep=1.0 / 1920.0))
    sub_backend = _load_backend(_freefall_scene(timestep=1.0 / 240.0))
    try:
        fine_result = fine.reset()
        for _ in range(8):
            fine_result = fine.step()
        runner = MuJoCoSubstepRunner(sub_backend, macro_timestep=1.0 / 240.0)
        runner.reset()
        sub_result = runner.step(substep_count=8)

        fine_state = fine_result.get_body_state(runtime_id)
        sub_state = sub_result.simulation_result.get_body_state(runtime_id)
        assert sub_result.simulation_result.time == pytest.approx(fine_result.time, abs=1.0e-12)
        assert sub_state.position == pytest.approx(fine_state.position, abs=1.0e-12)
        assert sub_state.linear_velocity == pytest.approx(fine_state.linear_velocity, abs=1.0e-12)
        assert sub_state.angular_velocity == pytest.approx(fine_state.angular_velocity, abs=1.0e-12)
        assert sub_result.substep_timestep == pytest.approx(1.0 / 1920.0)
    finally:
        fine.close()
        sub_backend.close()


def test_substep_counts_track_macro_and_physics_steps() -> None:
    backend = _load_backend(_freefall_scene(timestep=1.0 / 240.0))
    try:
        runner = MuJoCoSubstepRunner(backend, macro_timestep=1.0 / 240.0)
        runner.reset()

        first = runner.step(substep_count=16)
        second = runner.step(substep_count=4)

        assert first.macro_step_index == 1
        assert first.physics_step_count == 16
        assert first.simulation_result.step_index == 16
        assert second.macro_step_index == 2
        assert second.physics_step_count == 20
        assert second.simulation_result.step_index == 20
        assert runner.reset().step_index == 0
        assert runner.macro_step_index == 0
        assert runner.physics_step_count == 0
    finally:
        backend.close()


def test_substepped_sphere_drop_matches_fixed_fine_better_than_coarse() -> None:
    runtime_id = "sphere_01/sphere"
    total_macro_steps = 240
    coarse = _load_backend(_sphere_drop_scene(timestep=1.0 / 240.0))
    fine = _load_backend(_sphere_drop_scene(timestep=1.0 / 3840.0))
    sub_backend = _load_backend(_sphere_drop_scene(timestep=1.0 / 240.0))
    try:
        coarse_measurement, coarse_state = _measure_backend(coarse, runtime_id, steps=total_macro_steps)
        fine_measurement, fine_state = _measure_backend(fine, runtime_id, steps=total_macro_steps * 16)
        sub_measurement, sub_state = _measure_runner(
            sub_backend,
            runtime_id,
            macro_steps=total_macro_steps,
            substep_count=16,
        )

        fine_duration_state = _run_backend_steps(
            _sphere_drop_scene(timestep=1.0 / 3840.0),
            runtime_id,
            steps=total_macro_steps * 16,
        )
        sub_duration_state = _run_runner_steps(
            _sphere_drop_scene(timestep=1.0 / 240.0),
            runtime_id,
            macro_steps=total_macro_steps,
            substep_count=16,
        )

        assert sub_measurement.outcome is fine_measurement.outcome
        assert sub_duration_state.position == pytest.approx(fine_duration_state.position, abs=4.0e-3)
        assert sub_duration_state.linear_velocity == pytest.approx(fine_duration_state.linear_velocity, abs=4.0e-3)
        assert sub_measurement.measured_restitution == pytest.approx(fine_measurement.measured_restitution, abs=0.01)
        assert sub_measurement.maximum_penetration_depth == pytest.approx(
            fine_measurement.maximum_penetration_depth,
            abs=2.0e-3,
        )
        assert abs(sub_measurement.maximum_penetration_depth - fine_measurement.maximum_penetration_depth) < abs(
            coarse_measurement.maximum_penetration_depth - fine_measurement.maximum_penetration_depth
        )
        assert fine_measurement.contact_duration_seconds is not None
        assert sub_measurement.contact_duration_seconds is not None
        assert sub_measurement.contact_duration_seconds == pytest.approx(
            fine_measurement.contact_duration_seconds,
            abs=1.0 / 240.0,
        )
    finally:
        coarse.close()
        fine.close()
        sub_backend.close()


def test_resting_contact_substeps_keep_time_finite_and_stable() -> None:
    runtime_id = "sphere_01/sphere"
    scene = _sphere_drop_scene(timestep=1.0 / 240.0, initial_height=0.1, solref=(0.02, 1.0))
    first_backend = _load_backend(scene)
    second_backend = _load_backend(scene)
    try:
        first_runner = MuJoCoSubstepRunner(first_backend, macro_timestep=1.0 / 240.0)
        second_runner = MuJoCoSubstepRunner(second_backend, macro_timestep=1.0 / 240.0)
        first_runner.reset()
        second_runner.reset()
        first = None
        second = None
        for _ in range(120):
            first = first_runner.step(substep_count=1)
            second = second_runner.step(substep_count=8)

        assert first is not None and second is not None
        assert first.simulation_result.time == pytest.approx(0.5)
        assert second.simulation_result.time == pytest.approx(0.5)
        first_state = first.simulation_result.get_body_state(runtime_id)
        second_state = second.simulation_result.get_body_state(runtime_id)
        assert first_state.position[2] == pytest.approx(0.1, abs=0.03)
        assert second_state.position[2] == pytest.approx(0.1, abs=0.03)
        assert all(abs(value) < 10.0 for value in (*first_state.linear_velocity, *second_state.linear_velocity))
    finally:
        first_backend.close()
        second_backend.close()


def _measure_backend(backend: MuJoCoBackend, runtime_body_id: str, *, steps: int):
    measurement = measure_restitution(backend, runtime_body_id, max_steps=steps, characteristic_length=0.1)
    state = backend.get_body_state(runtime_body_id)
    return measurement, state


def _run_backend_steps(scene, runtime_body_id: str, *, steps: int):
    backend = _load_backend(scene)
    try:
        result = backend.reset()
        for _ in range(steps):
            result = backend.step()
        return result.get_body_state(runtime_body_id)
    finally:
        backend.close()


def _run_runner_steps(scene, runtime_body_id: str, *, macro_steps: int, substep_count: int):
    backend = _load_backend(scene)
    try:
        runner = MuJoCoSubstepRunner(backend, macro_timestep=1.0 / 240.0)
        result = runner.reset()
        for _ in range(macro_steps):
            result = runner.step(substep_count=substep_count).simulation_result
        return result.get_body_state(runtime_body_id)
    finally:
        backend.close()


def _measure_runner(
    backend: MuJoCoBackend,
    runtime_body_id: str,
    *,
    macro_steps: int,
    substep_count: int,
):
    runner = MuJoCoSubstepRunner(backend, macro_timestep=1.0 / 240.0)
    result = runner.reset()
    samples = [result]
    for _ in range(macro_steps):
        result = runner.step(substep_count=substep_count, substep_callback=samples.append).simulation_result
    measurement = _measure_from_results(
        tuple(samples),
        runtime_body_id,
        characteristic_length=0.1,
        timestep=1.0 / 3840.0,
    )
    return measurement, result.get_body_state(runtime_body_id)


def _measure_from_results(results, runtime_body_id: str, *, characteristic_length: float, timestep: float):
    last_downward_speed = 0.0
    impact_speed = 0.0
    contact_start_step = None
    contact_end_step = None
    last_contact_step = None
    maximum_penetration_depth = 0.0
    observed_contact_steps = 0
    for result in results:
        state = result.get_body_state(runtime_body_id)
        if contact_start_step is None and state.linear_velocity[2] < 0.0:
            last_downward_speed = -state.linear_velocity[2]
        contacts = tuple(c for c in result.contacts if c.body_a == runtime_body_id or c.body_b == runtime_body_id)
        if contacts:
            if contact_start_step is None:
                contact_start_step = result.step_index
                impact_speed = last_downward_speed
            last_contact_step = result.step_index
            observed_contact_steps += 1
            maximum_penetration_depth = max(maximum_penetration_depth, max(c.penetration_depth for c in contacts))
        elif contact_start_step is not None and contact_end_step is None and last_contact_step is not None:
            contact_end_step = last_contact_step
        if contact_end_step is not None and state.linear_velocity[2] > 1.0e-6:
            rebound = state.linear_velocity[2]
            from physical_simulation.evaluation.contact_calibration import RestitutionMeasurement

            duration_steps = max(0, contact_end_step - contact_start_step + 1)
            return RestitutionMeasurement(
                runtime_body_id=runtime_body_id,
                outcome=RestitutionOutcome.REBOUNDED,
                impact_speed=impact_speed,
                rebound_speed=rebound,
                measured_restitution=rebound / impact_speed,
                contact_start_step=contact_start_step,
                contact_end_step=contact_end_step,
                contact_duration_steps=duration_steps,
                contact_duration_seconds=duration_steps * timestep,
                maximum_penetration_depth=maximum_penetration_depth,
                normalized_penetration=maximum_penetration_depth / characteristic_length,
                observed_contact_steps=observed_contact_steps,
            )
    raise AssertionError("expected rebound measurement in sampled results")
