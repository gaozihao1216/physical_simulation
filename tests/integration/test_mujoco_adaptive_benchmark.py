from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.evaluation import (
    BenchmarkMode,
    BenchmarkValidity,
    SpherePlaneBenchmarkCase,
    SphereSphereBenchmarkCase,
    improvement_rates,
    run_contact_benchmark,
)
from physical_simulation.mujoco import SubstepRecommendationConfig


def _by_mode(dataset, case_id):
    return {result.mode: result for result in dataset.results if result.case_id == case_id}


def test_regression_case_detects_fixed_coarse_nonphysical_rebound() -> None:
    case = SpherePlaneBenchmarkCase(
        case_id="regression",
        initial_height=1.0,
        macro_timestep=1.0 / 240.0,
        solref=(0.005, 0.3),
        total_simulation_time=1.0,
    )
    dataset = run_contact_benchmark(
        (case,),
        recommendation=SubstepRecommendationConfig(maximum_substeps=16),
    )
    results = _by_mode(dataset, "regression")
    coarse = results[BenchmarkMode.FIXED_COARSE]
    fine = results[BenchmarkMode.FIXED_FINE]
    adaptive = results[BenchmarkMode.ADAPTIVE]
    comparison = dataset.comparisons[0]

    assert coarse.validity is BenchmarkValidity.NONPHYSICAL_REBOUND
    assert coarse.restitution is not None and coarse.restitution > 1.05
    assert fine.restitution == pytest.approx(0.4328, rel=0.08)
    assert adaptive.restitution == pytest.approx(fine.restitution, abs=0.03)
    assert adaptive.physics_step_count < fine.physics_step_count * 0.25
    assert comparison.adaptive_step_ratio < 0.25
    assert comparison.adaptive_improves_restitution is True
    assert comparison.adaptive_improves_penetration is True


def test_sphere_plane_benchmark_improves_majority_of_valid_cases() -> None:
    cases = (
        SpherePlaneBenchmarkCase(
            case_id="drop_a",
            initial_height=0.7,
            macro_timestep=1.0 / 240.0,
            solref=(0.02, 0.3),
            total_simulation_time=0.9,
        ),
        SpherePlaneBenchmarkCase(
            case_id="drop_b",
            initial_height=1.0,
            macro_timestep=1.0 / 120.0,
            solref=(0.01, 0.3),
            total_simulation_time=1.0,
        ),
    )
    dataset = run_contact_benchmark(cases, recommendation=SubstepRecommendationConfig(maximum_substeps=16))
    rates = improvement_rates(dataset.comparisons)

    assert len(dataset.results) == 6
    assert rates["restitution"] >= 0.5
    assert rates["penetration"] >= 0.5
    for comparison in dataset.comparisons:
        assert comparison.adaptive_step_ratio < 1.0
        assert comparison.adaptive_step_saving > 0.0


def test_sphere_sphere_benchmark_runs_all_modes() -> None:
    case = SphereSphereBenchmarkCase(
        case_id="headon",
        macro_timestep=1.0 / 240.0,
        solref=(0.01, 0.3),
        total_simulation_time=0.5,
    )
    dataset = run_contact_benchmark((case,), recommendation=SubstepRecommendationConfig(maximum_substeps=16))
    results = _by_mode(dataset, "headon")

    assert set(results) == {BenchmarkMode.FIXED_COARSE, BenchmarkMode.FIXED_FINE, BenchmarkMode.ADAPTIVE}
    assert results[BenchmarkMode.FIXED_FINE].outcome.value in {"rebounded", "timeout"}
    assert results[BenchmarkMode.ADAPTIVE].physics_step_count < results[BenchmarkMode.FIXED_FINE].physics_step_count
    assert results[BenchmarkMode.ADAPTIVE].adaptive_max_substep_count is not None
    assert dataset.comparisons[0].adaptive_step_ratio < 1.0


def test_benchmark_repeat_is_deterministic_except_wall_time() -> None:
    case = SpherePlaneBenchmarkCase(
        case_id="deterministic",
        initial_height=0.4,
        macro_timestep=1.0 / 240.0,
        solref=(0.02, 0.5),
        total_simulation_time=0.7,
    )
    first = run_contact_benchmark((case,), recommendation=SubstepRecommendationConfig(maximum_substeps=16))
    second = run_contact_benchmark((case,), recommendation=SubstepRecommendationConfig(maximum_substeps=16))

    for left, right in zip(first.results, second.results):
        assert left.mode is right.mode
        assert left.validity is right.validity
        assert left.outcome is right.outcome
        assert left.physics_step_count == right.physics_step_count
        assert left.restitution == pytest.approx(right.restitution) if left.restitution is not None else right.restitution is None
        assert left.maximum_penetration == pytest.approx(right.maximum_penetration)
