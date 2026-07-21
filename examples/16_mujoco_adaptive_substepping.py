"""Demonstrate explicit-candidate adaptive MuJoCo substepping."""

from __future__ import annotations

from dataclasses import dataclass, replace

from physical_simulation.assets import Transform, create_ground, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import RestitutionOutcome, measure_restitution
from physical_simulation.evaluation.contact_calibration import RestitutionMeasurement
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
MACRO_TIMESTEP = 1.0 / 240.0


@dataclass(frozen=True)
class RunSummary:
    label: str
    measurement: RestitutionMeasurement
    final_position_z: float
    final_velocity_z: float
    mj_step_count: int


def _params() -> MuJoCoContactSolverParams:
    return MuJoCoContactSolverParams(solref=(0.005, 0.3), solimp=(0.9, 0.9, 0.001, 0.5, 2.0))


def _body_with_params(body, params):
    return replace(body, colliders=tuple(replace(collider, mujoco_contact_params=params) for collider in body.colliders))


def _scene(*, timestep: float, params: MuJoCoContactSolverParams):
    ground = create_single_body_asset(
        asset_id="ground_asset",
        body=_body_with_params(create_ground("ground"), params),
    )
    sphere = create_single_body_asset(
        asset_id="sphere_asset",
        body=_body_with_params(create_sphere("sphere", SPHERE_RADIUS, mass=1.0), params),
    )
    return create_scene(
        scene_id="mujoco_adaptive_substepping",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("sphere_01", sphere, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=timestep,
    )


def _candidate(params: MuJoCoContactSolverParams) -> SpherePlaneAdaptiveCandidate:
    return SpherePlaneAdaptiveCandidate(
        "sphere_ground",
        RUNTIME_BODY_ID,
        SPHERE_RADIUS,
        AnalyticPlane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)),
        params,
    )


def _run_fixed(label: str, *, timestep: float, steps: int, params: MuJoCoContactSolverParams) -> RunSummary:
    backend = MuJoCoBackend()
    backend.load_scene(_scene(timestep=timestep, params=params))
    try:
        measurement = measure_restitution(backend, RUNTIME_BODY_ID, max_steps=steps, characteristic_length=SPHERE_RADIUS)
        result = backend.reset()
        for _ in range(steps):
            result = backend.step()
        state = result.get_body_state(RUNTIME_BODY_ID)
        return RunSummary(label, measurement, state.position[2], state.linear_velocity[2], steps)
    finally:
        backend.close()


def _run_adaptive(*, params: MuJoCoContactSolverParams, steps: int) -> RunSummary:
    backend = MuJoCoBackend()
    backend.load_scene(_scene(timestep=MACRO_TIMESTEP, params=params))
    try:
        runner = AdaptiveMuJoCoRunner(
            backend,
            candidates=(_candidate(params),),
            config=AdaptiveSubstepConfig(
                recommendation=SubstepRecommendationConfig(maximum_substeps=16),
                resting_window_macro_steps=3,
                separating_hold_macro_steps=1,
            ),
        )
        samples = [runner.reset()]
        previous_state = ContactMotionState.FREE
        print("time,state transition,candidate,time_to_contact,solver_timescale,substeps,substep_dt,active_contact")
        for _ in range(steps):
            adaptive = runner.step()
            decision = adaptive.decision
            samples.extend(adaptive.substep_results)
            if not adaptive.substep_results:
                samples.append(adaptive.advance_result.simulation_result)
            if decision.state_after is not previous_state or decision.substep_count > 1 or decision.active_contact_observed:
                print(
                    f"{adaptive.advance_result.simulation_result.time:.6f},"
                    f"{decision.state_before.value}->{decision.state_after.value},"
                    f"{decision.selected_candidate_id},"
                    f"{_fmt(decision.prediction.time_to_contact if decision.prediction else None)},"
                    f"{_fmt(decision.solver_estimate.timescale.characteristic_timescale if decision.solver_estimate else None)},"
                    f"{decision.substep_count},"
                    f"{decision.actual_substep_timestep:.9f},"
                    f"{decision.active_contact_observed}"
                )
            previous_state = decision.state_after
        measurement = _measure_macro_samples(samples)
        final_state = samples[-1].get_body_state(RUNTIME_BODY_ID)
        return RunSummary("Adaptive", measurement, final_state.position[2], final_state.linear_velocity[2], runner.physics_step_count)
    finally:
        backend.close()


def _measure_macro_samples(samples: list) -> RestitutionMeasurement:
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
            duration = contact_end - contact_start + 1
            return RestitutionMeasurement(
                runtime_body_id=RUNTIME_BODY_ID,
                outcome=RestitutionOutcome.REBOUNDED,
                impact_speed=impact,
                rebound_speed=state.linear_velocity[2],
                measured_restitution=state.linear_velocity[2] / impact,
                contact_start_step=contact_start,
                contact_end_step=contact_end,
                contact_duration_steps=duration,
                contact_duration_seconds=None if contact_start_time is None or contact_end_time is None else contact_end_time - contact_start_time,
                maximum_penetration_depth=max_penetration,
                normalized_penetration=max_penetration / SPHERE_RADIUS,
                observed_contact_steps=observed,
            )
    raise RuntimeError("adaptive run did not rebound in macro samples")


def main() -> None:
    params = _params()
    coarse = _run_fixed("Fixed coarse", timestep=MACRO_TIMESTEP, steps=240, params=params)
    fine = _run_fixed("Fixed fine", timestep=MACRO_TIMESTEP / 16.0, steps=3840, params=params)
    adaptive = _run_adaptive(params=params, steps=240)
    print()
    print("label,restitution,penetration,contact_duration,final_z,final_vz,mj_step_count")
    for summary in (coarse, fine, adaptive):
        print(
            f"{summary.label},"
            f"{_fmt(summary.measurement.measured_restitution)},"
            f"{summary.measurement.maximum_penetration_depth:.6f},"
            f"{_fmt(summary.measurement.contact_duration_seconds)},"
            f"{summary.final_position_z:.6f},"
            f"{summary.final_velocity_z:.6f},"
            f"{summary.mj_step_count}"
        )
    print(f"adaptive fine-step ratio: {adaptive.mj_step_count / fine.mj_step_count:.6f}")


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
