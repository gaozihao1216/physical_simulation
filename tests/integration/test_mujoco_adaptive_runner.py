from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_ground, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import RestitutionOutcome, measure_restitution
from physical_simulation.mujoco import (
    AdaptiveMuJoCoRunner,
    AdaptiveSubstepConfig,
    AnalyticPlane,
    ContactMotionState,
    MuJoCoContactSolverParams,
    SpherePlaneAdaptiveCandidate,
    SubstepRecommendationConfig,
)
from physical_simulation.scene import AssetInstanceSpec, create_scene

RUNTIME_BODY_ID = "sphere_01/sphere"
SPHERE_RADIUS = 0.1


def _params(solref=(0.005, 0.3)):
    return MuJoCoContactSolverParams(solref=solref, solimp=(0.9, 0.9, 0.001, 0.5, 2.0))


def _body_with_params(body, params):
    return replace(body, colliders=tuple(replace(collider, mujoco_contact_params=params) for collider in body.colliders))


def _scene(*, timestep=1.0 / 240.0, params=None, height=1.0):
    selected_params = params or _params()
    ground = create_single_body_asset(
        asset_id="ground_asset",
        body=_body_with_params(create_ground("ground"), selected_params),
    )
    sphere = create_single_body_asset(
        asset_id="sphere_asset",
        body=_body_with_params(create_sphere("sphere", SPHERE_RADIUS, mass=1.0, create_visual=False), selected_params),
    )
    return create_scene(
        scene_id="adaptive_runner",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("sphere_01", sphere, Transform(position=(0.0, 0.0, height))),
        ),
        timestep=timestep,
    )


def _candidate(params=None):
    return SpherePlaneAdaptiveCandidate(
        "sphere_ground",
        RUNTIME_BODY_ID,
        SPHERE_RADIUS,
        AnalyticPlane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)),
        params or _params(),
    )


def _runner(*, params=None, height=1.0, maximum_substeps=64, resting_window=3):
    backend = MuJoCoBackend()
    selected_params = params or _params()
    backend.load_scene(_scene(params=selected_params, height=height))
    runner = AdaptiveMuJoCoRunner(
        backend,
        candidates=(_candidate(selected_params),),
        config=AdaptiveSubstepConfig(
            recommendation=SubstepRecommendationConfig(maximum_substeps=maximum_substeps),
            resting_window_macro_steps=resting_window,
            separating_hold_macro_steps=1,
        ),
    )
    return backend, runner


def test_far_from_contact_stays_free_with_single_substep() -> None:
    backend, runner = _runner()
    try:
        runner.reset()
        result = runner.step()

        assert result.decision.state_before is ContactMotionState.FREE
        assert result.decision.state_after is ContactMotionState.FREE
        assert result.decision.substep_count == 1
        assert result.decision.selected_candidate_id is None
    finally:
        backend.close()


def test_prediction_enters_approaching_and_uses_substeps() -> None:
    backend, runner = _runner()
    try:
        runner.reset()
        decisions = [runner.step().decision for _ in range(120)]
        approaching = next(decision for decision in decisions if decision.state_after is ContactMotionState.APPROACHING)

        assert approaching.substep_count > 1
        assert approaching.prediction is not None
        assert approaching.solver_estimate is not None
        assert approaching.selected_candidate_id == "sphere_ground"
    finally:
        backend.close()


def test_impacting_continues_fine_substeps_and_rebound_separates_to_free() -> None:
    backend, runner = _runner()
    try:
        runner.reset()
        decisions = [runner.step().decision for _ in range(220)]
        impacting = [decision for decision in decisions if decision.state_after is ContactMotionState.IMPACTING]
        separating = [decision for decision in decisions if decision.state_after is ContactMotionState.SEPARATING]
        free_after_separating = any(
            previous.state_after is ContactMotionState.SEPARATING and current.state_after is ContactMotionState.FREE
            for previous, current in zip(decisions, decisions[1:])
        )

        assert impacting
        assert max(decision.substep_count for decision in impacting) > 1
        assert separating
        assert free_after_separating
    finally:
        backend.close()


def test_resting_contact_recovers_to_macro_timestep() -> None:
    params = _params(solref=(0.02, 1.0))
    backend, runner = _runner(params=params, height=0.1, resting_window=2)
    try:
        runner.reset()
        decisions = [runner.step().decision for _ in range(80)]
        resting = [decision for decision in decisions if decision.state_after is ContactMotionState.RESTING]

        assert resting
        assert decisions[-1].state_after is ContactMotionState.RESTING
        assert decisions[-1].substep_count == 1
    finally:
        backend.close()


def test_maximum_substeps_clips_adaptive_recommendation() -> None:
    backend, runner = _runner(maximum_substeps=4)
    try:
        runner.reset()
        decisions = [runner.step().decision for _ in range(120)]
        clipped = [decision for decision in decisions if decision.solver_estimate is not None]

        assert clipped
        assert max(decision.substep_count for decision in clipped) == 4
        assert any(decision.solver_estimate.recommendation.limited_by_maximum_substeps for decision in clipped)
    finally:
        backend.close()


def test_multiple_candidates_select_smallest_actual_substep_timestep() -> None:
    fast_params = _params(solref=(0.005, 0.3))
    slow_params = _params(solref=(0.02, 0.5))
    backend = MuJoCoBackend()
    backend.load_scene(_scene(params=fast_params))
    try:
        runner = AdaptiveMuJoCoRunner(
            backend,
            candidates=(
                SpherePlaneAdaptiveCandidate(
                    "slow",
                    RUNTIME_BODY_ID,
                    SPHERE_RADIUS,
                    AnalyticPlane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)),
                    slow_params,
                ),
                SpherePlaneAdaptiveCandidate(
                    "fast",
                    RUNTIME_BODY_ID,
                    SPHERE_RADIUS,
                    AnalyticPlane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)),
                    fast_params,
                ),
            ),
            config=AdaptiveSubstepConfig(recommendation=SubstepRecommendationConfig(maximum_substeps=64)),
        )
        runner.reset()
        decisions = [runner.step().decision for _ in range(120)]
        selected = next(decision for decision in decisions if decision.solver_estimate is not None)

        assert selected.selected_candidate_id == "fast"
        assert selected.substep_count > 1
    finally:
        backend.close()


def test_reset_is_deterministic_and_clears_counts() -> None:
    backend, runner = _runner()
    try:
        first_reset = runner.reset()
        first = runner.step()
        runner.reset()
        second = runner.step()

        assert first_reset.step_index == 0
        assert first.decision == second.decision
        assert first.advance_result.macro_step_index == second.advance_result.macro_step_index == 1
        assert first.advance_result.physics_step_count == second.advance_result.physics_step_count
    finally:
        backend.close()


def test_adaptive_is_closer_to_fixed_fine_than_coarse_with_fewer_steps() -> None:
    params = _params()
    coarse_measurement, coarse_state, coarse_steps = _run_fixed(timestep=1.0 / 240.0, steps=240, params=params)
    fine_measurement, fine_state, fine_steps = _run_fixed(timestep=1.0 / 3840.0, steps=3840, params=params)
    adaptive_measurement, adaptive_state, adaptive_steps = _run_adaptive(steps=240, params=params)

    assert adaptive_measurement.outcome is RestitutionOutcome.REBOUNDED
    assert adaptive_steps < fine_steps
    assert adaptive_steps > coarse_steps
    assert abs(adaptive_measurement.maximum_penetration_depth - fine_measurement.maximum_penetration_depth) < abs(
        coarse_measurement.maximum_penetration_depth - fine_measurement.maximum_penetration_depth
    )
    assert abs(adaptive_state.position[2] - fine_state.position[2]) < abs(coarse_state.position[2] - fine_state.position[2])
    assert abs(adaptive_state.linear_velocity[2] - fine_state.linear_velocity[2]) < abs(
        coarse_state.linear_velocity[2] - fine_state.linear_velocity[2]
    )


def _run_fixed(*, timestep: float, steps: int, params: MuJoCoContactSolverParams):
    backend = MuJoCoBackend()
    backend.load_scene(_scene(timestep=timestep, params=params))
    try:
        measurement = measure_restitution(backend, RUNTIME_BODY_ID, max_steps=steps, characteristic_length=SPHERE_RADIUS)
        result = backend.reset()
        for _ in range(steps):
            result = backend.step()
        return measurement, result.get_body_state(RUNTIME_BODY_ID), steps
    finally:
        backend.close()


def _run_adaptive(*, steps: int, params: MuJoCoContactSolverParams):
    backend, runner = _runner(params=params)
    samples = []
    try:
        result = runner.reset()
        samples.append(result)
        for _ in range(steps):
            adaptive = runner.step()
            result = adaptive.advance_result.simulation_result
            samples.extend(adaptive.substep_results or (result,))
        measurement = _measure_samples(samples)
        return measurement, result.get_body_state(RUNTIME_BODY_ID), runner.physics_step_count
    finally:
        backend.close()


def _measure_samples(samples):
    last_downward = 0.0
    impact = 0.0
    contact_start = None
    contact_start_time = None
    contact_end = None
    contact_end_time = None
    last_contact = None
    last_contact_time = None
    max_penetration = 0.0
    observed = 0
    for sample in samples:
        state = sample.get_body_state(RUNTIME_BODY_ID)
        if contact_start is None and state.linear_velocity[2] < 0.0:
            last_downward = -state.linear_velocity[2]
        contacts = tuple(c for c in sample.contacts if c.body_a == RUNTIME_BODY_ID or c.body_b == RUNTIME_BODY_ID)
        if contacts:
            if contact_start is None:
                contact_start = sample.step_index
                contact_start_time = sample.time
                impact = last_downward
            last_contact = sample.step_index
            last_contact_time = sample.time
            observed += 1
            max_penetration = max(max_penetration, max(c.penetration_depth for c in contacts))
        elif contact_start is not None and contact_end is None and last_contact is not None:
            contact_end = last_contact
            contact_end_time = last_contact_time
        if contact_end is not None and state.linear_velocity[2] > 1.0e-6:
            from physical_simulation.evaluation.contact_calibration import RestitutionMeasurement

            duration_steps = contact_end - contact_start + 1
            return RestitutionMeasurement(
                runtime_body_id=RUNTIME_BODY_ID,
                outcome=RestitutionOutcome.REBOUNDED,
                impact_speed=impact,
                rebound_speed=state.linear_velocity[2],
                measured_restitution=state.linear_velocity[2] / impact,
                contact_start_step=contact_start,
                contact_end_step=contact_end,
                contact_duration_steps=duration_steps,
                contact_duration_seconds=None if contact_start_time is None or contact_end_time is None else contact_end_time - contact_start_time,
                maximum_penetration_depth=max_penetration,
                normalized_penetration=max_penetration / SPHERE_RADIUS,
                observed_contact_steps=observed,
            )
    raise AssertionError("expected rebound in adaptive samples")
