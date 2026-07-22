"""Demonstrate automatic adaptive prediction candidate construction."""

from __future__ import annotations

from collections import Counter

from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation.contact_benchmark import (
    BenchmarkMode,
    SpherePlaneBenchmarkCase,
    SphereSphereBenchmarkCase,
    _apply_initial_velocity,
    _macro_steps,
    _measure_samples,
    _scene_for_case,
    run_contact_benchmark,
)
from physical_simulation.mujoco import (
    AdaptiveMuJoCoRunner,
    AdaptiveSubstepConfig,
    SpherePlaneAdaptiveCandidate,
    SphereSphereAdaptiveCandidate,
    build_adaptive_prediction_candidates,
)


def main() -> None:
    cases = (
        SpherePlaneBenchmarkCase("auto_candidate_sphere_plane", 0.4, 1.0 / 240.0, (0.02, 0.5), total_simulation_time=0.5),
        SphereSphereBenchmarkCase("auto_candidate_sphere_sphere", 1.0 / 240.0, (0.01, 0.3), total_simulation_time=0.35),
    )
    for case in cases:
        print(f"\ncase: {case.case_id}")
        benchmark = run_contact_benchmark((case,))
        measurement, physics_steps, state_counts, build = _run_auto(case)
        fine = next(result for result in benchmark.results if result.mode is BenchmarkMode.FIXED_FINE)
        step_saving = 1.0 - physics_steps / fine.physics_step_count
        print(f"inspected collider count: {build.inspected_collider_count}")
        print(f"eligible pair count: {build.eligible_pair_count}")
        print(f"generated candidate count: {build.generated_candidate_count}")
        for candidate in build.candidates:
            _print_candidate(candidate)
        print(f"skip diagnostics summary: {dict(Counter(item.reason for item in build.diagnostics if item.status.value != 'generated'))}")
        print(f"state transitions: {dict(state_counts)}")
        print(f"restitution: {measurement.measured_restitution}")
        print(f"penetration: {measurement.maximum_penetration_depth}")
        print(f"physics step count: {physics_steps}")
        print(f"step saving vs fixed fine: {step_saving:.6f}")


def _run_auto(case):
    scene = _scene_for_case(case, timestep=case.macro_timestep)
    backend = MuJoCoBackend()
    try:
        backend.load_scene(scene)
        _apply_initial_velocity(case, backend, update_initial=True)
        build = build_adaptive_prediction_candidates(scene=scene, backend=backend)
        runner = AdaptiveMuJoCoRunner(
            backend,
            candidates=build.candidates,
            config=AdaptiveSubstepConfig(macro_timestep=case.macro_timestep),
        )
        samples = [runner.reset()]
        states = Counter()
        for _ in range(_macro_steps(case)):
            step = runner.step()
            states[step.decision.state_after.value] += 1
            samples.extend(step.substep_results or (step.advance_result.simulation_result,))
        return _measure_samples(case, samples), runner.physics_step_count, states, build
    finally:
        backend.close()


def _print_candidate(candidate) -> None:
    if isinstance(candidate, SpherePlaneAdaptiveCandidate):
        print(
            "candidate: "
            f"id={candidate.candidate_id}, type=sphere_plane, sphere={candidate.sphere_runtime_body_id}, "
            f"radius={candidate.sphere_radius}, plane_point={candidate.plane.point}, "
            f"plane_normal={candidate.plane.normal}, solref={candidate.contact_params.solref}, "
            f"solimp={candidate.contact_params.solimp}"
        )
    elif isinstance(candidate, SphereSphereAdaptiveCandidate):
        print(
            "candidate: "
            f"id={candidate.candidate_id}, type=sphere_sphere, body_a={candidate.body_a_id}, "
            f"radius_a={candidate.radius_a}, body_b={candidate.body_b_id}, radius_b={candidate.radius_b}, "
            f"solref={candidate.contact_params.solref}, solimp={candidate.contact_params.solimp}"
        )


if __name__ == "__main__":
    main()

