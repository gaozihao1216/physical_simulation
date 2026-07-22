"""Integration tests for automatic adaptive candidate construction."""

from __future__ import annotations

from physical_simulation.assets import Transform, create_box, create_single_body_asset
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation.contact_benchmark import (
    BenchmarkMode,
    SpherePlaneBenchmarkCase,
    SphereSphereBenchmarkCase,
    compare_benchmark_results,
    run_contact_benchmark,
    _apply_initial_velocity,
    _macro_steps,
    _measure_samples,
    _scene_for_case,
)
from physical_simulation.mujoco import (
    AdaptiveMuJoCoRunner,
    AdaptiveSubstepConfig,
    ContactMotionState,
    build_adaptive_prediction_candidates,
)
from physical_simulation.scene import AssetInstanceSpec, create_scene


def test_auto_sphere_drop_candidate_runs_adaptive_close_to_fixed_fine() -> None:
    case = SpherePlaneBenchmarkCase("auto_drop", 0.4, 1.0 / 240.0, (0.02, 0.5), total_simulation_time=0.5)
    benchmark = run_contact_benchmark((case,))
    comparison = benchmark.comparisons[0]

    measurement, physics_steps, states, candidate_count = _run_auto_adaptive_case(case)

    assert candidate_count == 1
    assert ContactMotionState.APPROACHING in states or ContactMotionState.IMPACTING in states
    assert measurement.measured_restitution is not None
    fine = next(result for result in benchmark.results if result.mode is BenchmarkMode.FIXED_FINE)
    coarse_error = comparison.coarse_restitution_error
    adaptive_error = abs(measurement.measured_restitution - fine.restitution)
    assert coarse_error is not None
    assert adaptive_error < coarse_error
    assert physics_steps < fine.physics_step_count


def test_auto_sphere_sphere_candidate_improves_over_fixed_coarse() -> None:
    case = SphereSphereBenchmarkCase("auto_sphere_sphere", 1.0 / 240.0, (0.01, 0.3), total_simulation_time=0.35)
    benchmark = run_contact_benchmark((case,))
    comparison = benchmark.comparisons[0]

    measurement, _physics_steps, states, candidate_count = _run_auto_adaptive_case(case)
    fine = next(result for result in benchmark.results if result.mode is BenchmarkMode.FIXED_FINE)
    coarse = next(result for result in benchmark.results if result.mode is BenchmarkMode.FIXED_COARSE)

    assert candidate_count == 1
    assert ContactMotionState.APPROACHING in states or ContactMotionState.IMPACTING in states
    assert measurement.measured_restitution is not None
    assert abs(measurement.measured_restitution - fine.restitution) < abs(coarse.restitution - fine.restitution)
    assert comparison.adaptive_improves_penetration


def test_unsupported_box_box_scene_keeps_mujoco_contact_detection() -> None:
    scene = _box_drop_scene()
    backend = MuJoCoBackend()
    try:
        backend.load_scene(scene)
        candidates = build_adaptive_prediction_candidates(scene=scene, backend=backend)
        assert candidates.generated_candidate_count == 0
        assert candidates.skipped_unsupported_geometry_count == 1
        backend.reset()
        observed_contact = False
        for _ in range(180):
            result = backend.step()
            observed_contact = observed_contact or bool(result.contacts)
        assert observed_contact
    finally:
        backend.close()


def _run_auto_adaptive_case(case):
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
        states = []
        for _ in range(_macro_steps(case)):
            step = runner.step()
            states.append(step.decision.state_after)
            samples.extend(step.substep_results or (step.advance_result.simulation_result,))
        return _measure_samples(case, samples), runner.physics_step_count, tuple(states), build.generated_candidate_count
    finally:
        backend.close()


def _box_drop_scene():
    ground = create_box("ground", (2.0, 2.0, 0.1), body_type="static", transform=Transform(position=(0, 0, -0.05)), create_visual=False)
    box = create_box("box", (0.2, 0.2, 0.2), mass=1.0, create_visual=False)
    return create_scene(
        scene_id="box_drop_unsupported",
        instances=(
            AssetInstanceSpec("ground_01", create_single_body_asset(asset_id="ground_asset", body=ground), fixed_base=True),
            AssetInstanceSpec("box_01", create_single_body_asset(asset_id="box_asset", body=box), Transform(position=(0, 0, 0.8))),
        ),
        timestep=1.0 / 240.0,
    )

