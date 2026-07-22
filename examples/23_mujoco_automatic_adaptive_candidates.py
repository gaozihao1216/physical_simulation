"""Demonstrate automatic adaptive prediction candidate construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from physical_simulation.assets import Transform, create_box, create_capsule, create_single_body_asset
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
    ConservativePrimitiveAdaptiveCandidate,
    SpherePlaneAdaptiveCandidate,
    SphereSphereAdaptiveCandidate,
    build_adaptive_prediction_candidates,
)
from physical_simulation.scene import AssetInstanceSpec, create_scene


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

    for label, scene in (
        ("box_box_conservative", _box_drop_scene()),
        ("capsule_box_conservative", _capsule_box_scene()),
        ("compound_box_conservative", _compound_scene()),
    ):
        print(f"\ncase: {label}")
        summary = _run_generic_auto(scene, macro_steps=180)
        build = summary["build"]
        print(f"inspected collider count: {build.inspected_collider_count}")
        print(f"eligible pair count: {build.eligible_pair_count}")
        print(f"generated candidate count: {build.generated_candidate_count}")
        for candidate in build.candidates:
            _print_candidate(candidate)
        print(f"skip diagnostics summary: {dict(Counter(item.reason for item in build.diagnostics if item.status.value != 'generated'))}")
        print(f"state transitions: {dict(summary['states'])}")
        print(f"contact observed: {summary['contact_observed']}")
        print(f"prediction-only macro steps before contact: {summary['prediction_only_steps']}")
        print(f"conservative misfire count: {summary['misfire_count']}")
        print(f"physics step count: {summary['physics_steps']}")
        print(f"step saving vs fixed fine: {summary['step_saving']:.6f}")


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


def _run_generic_auto(scene, *, macro_steps: int):
    backend = MuJoCoBackend()
    try:
        backend.load_scene(scene)
        build = build_adaptive_prediction_candidates(scene=scene, backend=backend)
        runner = AdaptiveMuJoCoRunner(
            backend,
            candidates=build.candidates,
            config=AdaptiveSubstepConfig(macro_timestep=scene.timestep),
        )
        runner.reset()
        states = Counter()
        predicted_steps = 0
        prediction_only_steps = 0
        contact_observed = False
        for _ in range(macro_steps):
            step = runner.step()
            states[step.decision.state_after.value] += 1
            selected = step.decision.selected_candidate_id is not None
            predicted_steps += int(selected)
            contact_now = step.decision.active_contact_observed or bool(step.advance_result.simulation_result.contacts)
            if selected and not contact_now and not contact_observed:
                prediction_only_steps += 1
            contact_observed = contact_observed or contact_now
        fixed_fine_steps = macro_steps * 16
        return {
            "build": build,
            "states": states,
            "contact_observed": contact_observed,
            "prediction_only_steps": prediction_only_steps,
            "misfire_count": 0 if contact_observed else predicted_steps,
            "physics_steps": runner.physics_step_count,
            "step_saving": 1.0 - runner.physics_step_count / fixed_fine_steps,
        }
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
    elif isinstance(candidate, ConservativePrimitiveAdaptiveCandidate):
        print(
            "candidate: "
            f"id={candidate.candidate_id}, type=conservative_primitive, body_a={candidate.body_a_id}, "
            f"radius_a={candidate.bounding_radius_a}, body_b={candidate.body_b_id}, "
            f"radius_b={candidate.bounding_radius_b}, solref={candidate.contact_params.solref}, "
            f"solimp={candidate.contact_params.solimp}"
        )


def _box_drop_scene():
    ground = create_box("ground", (2.0, 2.0, 0.1), body_type="static", transform=Transform(position=(0, 0, -0.05)), create_visual=False)
    box = create_box("box", (0.2, 0.2, 0.2), mass=1.0, create_visual=False)
    return create_scene(
        scene_id="auto_candidate_box_box",
        instances=(
            AssetInstanceSpec("ground_01", create_single_body_asset(asset_id="ground_asset", body=ground), fixed_base=True),
            AssetInstanceSpec("box_01", create_single_body_asset(asset_id="box_asset", body=box), Transform(position=(0, 0, 0.8))),
        ),
        timestep=1.0 / 240.0,
    )


def _capsule_box_scene():
    ground = create_box("ground", (2.0, 2.0, 0.1), body_type="static", transform=Transform(position=(0, 0, -0.05)), create_visual=False)
    capsule = create_capsule("capsule", 0.05, 0.2, mass=1.0, create_visual=False)
    return create_scene(
        scene_id="auto_candidate_capsule_box",
        instances=(
            AssetInstanceSpec("ground_01", create_single_body_asset(asset_id="ground_asset", body=ground), fixed_base=True),
            AssetInstanceSpec("capsule_01", create_single_body_asset(asset_id="capsule_asset", body=capsule), Transform(position=(0, 0, 0.8))),
        ),
        timestep=1.0 / 240.0,
    )


def _compound_scene():
    ground = create_box("ground", (2.0, 2.0, 0.1), body_type="static", transform=Transform(position=(0, 0, -0.05)), create_visual=False)
    body = create_box("compound", (0.2, 0.2, 0.2), mass=1.0, create_visual=False)
    body = replace(
        body,
        colliders=(
            body.colliders[0],
            replace(body.colliders[0], collider_id="compound_offset", local_transform=Transform(position=(0.25, 0, 0))),
        ),
    )
    return create_scene(
        scene_id="auto_candidate_compound",
        instances=(
            AssetInstanceSpec("ground_01", create_single_body_asset(asset_id="ground_asset", body=ground), fixed_base=True),
            AssetInstanceSpec("compound_01", create_single_body_asset(asset_id="compound_asset", body=body), Transform(position=(0, 0, 0.8))),
        ),
        timestep=1.0 / 240.0,
    )


if __name__ == "__main__":
    main()
