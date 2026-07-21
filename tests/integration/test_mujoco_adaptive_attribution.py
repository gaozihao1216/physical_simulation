from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.evaluation import (
    AdaptiveFailureReason,
    BenchmarkMode,
    ImprovementOutcome,
    ReferenceConvergenceConfig,
    ReferenceConvergenceStatus,
    SpherePlaneBenchmarkCase,
    SphereSphereBenchmarkCase,
    attribute_adaptive_failure,
    build_attribution_dataset,
    run_adaptive_diagnostic_trace,
    run_contact_benchmark,
    run_reference_convergence,
)
from physical_simulation.mujoco import AdaptiveSubstepConfig, SubstepRecommendationConfig


def _adaptive_result(dataset):
    return next(result for result in dataset.results if result.mode is BenchmarkMode.ADAPTIVE)


def test_good_sphere_plane_reference_converges_and_has_no_failure_reason() -> None:
    case = SpherePlaneBenchmarkCase("good", 1.0, 1.0 / 240.0, (0.02, 0.3), total_simulation_time=0.6)
    recommendation = SubstepRecommendationConfig(maximum_substeps=16)
    benchmark = run_contact_benchmark((case,), recommendation=recommendation)
    convergence = run_reference_convergence(case, recommendation=recommendation)
    trace = run_adaptive_diagnostic_trace(case, recommendation=recommendation)
    attribution = attribute_adaptive_failure(
        comparison=benchmark.comparisons[0],
        adaptive_result=_adaptive_result(benchmark),
        trace=trace,
        convergence=convergence,
        recommendation=recommendation,
    )

    assert convergence.overall_status is ReferenceConvergenceStatus.CONVERGED
    assert trace.total_contact_episode_count >= 1
    assert attribution.primary_reason is AdaptiveFailureReason.NONE
    assert attribution.restitution_outcome in {ImprovementOutcome.IMPROVED, ImprovementOutcome.BOTH_ACCEPTABLE}


def test_short_prediction_horizon_attributes_late_or_short_prediction() -> None:
    case = SpherePlaneBenchmarkCase("short", 1.0, 1.0 / 240.0, (0.02, 0.3), total_simulation_time=0.6)
    recommendation = SubstepRecommendationConfig(maximum_substeps=16)
    benchmark = run_contact_benchmark((case,), recommendation=recommendation)
    convergence = run_reference_convergence(case, recommendation=recommendation)
    trace = run_adaptive_diagnostic_trace(
        case,
        recommendation=recommendation,
        adaptive_config=AdaptiveSubstepConfig(
            macro_timestep=case.macro_timestep,
            prediction_horizon_multiplier=0.1,
            recommendation=recommendation,
        ),
    )
    attribution = attribute_adaptive_failure(
        comparison=benchmark.comparisons[0],
        adaptive_result=_adaptive_result(benchmark),
        trace=trace,
        convergence=convergence,
        recommendation=recommendation,
    )

    assert attribution.primary_reason in {
        AdaptiveFailureReason.LATE_PREDICTION,
        AdaptiveFailureReason.SHORT_PREDICTION_LEAD,
    }


def test_maximum_substeps_limited_is_reported() -> None:
    case = SpherePlaneBenchmarkCase("limited", 1.0, 1.0 / 240.0, (0.005, 0.3), total_simulation_time=1.0)
    recommendation = SubstepRecommendationConfig(maximum_substeps=2)
    benchmark = run_contact_benchmark((case,), recommendation=recommendation)
    convergence = run_reference_convergence(
        case,
        recommendation=recommendation,
        config=ReferenceConvergenceConfig(restitution_absolute_tolerance=0.0, penetration_absolute_tolerance=0.0),
    )
    trace = run_adaptive_diagnostic_trace(case, recommendation=recommendation)
    attribution = attribute_adaptive_failure(
        comparison=benchmark.comparisons[0],
        adaptive_result=_adaptive_result(benchmark),
        trace=trace,
        convergence=convergence,
        recommendation=recommendation,
    )

    assert any(episode.limited_by_maximum_substeps for episode in trace.episodes)
    assert AdaptiveFailureReason.MAX_SUBSTEPS_LIMITED in {attribution.primary_reason, *attribution.secondary_reasons}


def test_multiple_impacts_and_unresolved_reference_are_reported() -> None:
    case = SpherePlaneBenchmarkCase("multi", 0.4, 1.0 / 240.0, (0.02, 0.5), total_simulation_time=0.7)
    recommendation = SubstepRecommendationConfig(maximum_substeps=8)
    benchmark = run_contact_benchmark((case,), recommendation=recommendation)
    convergence = run_reference_convergence(
        case,
        recommendation=recommendation,
        config=ReferenceConvergenceConfig(restitution_absolute_tolerance=0.0, penetration_absolute_tolerance=0.0),
    )
    trace = run_adaptive_diagnostic_trace(case, recommendation=recommendation)
    attribution = attribute_adaptive_failure(
        comparison=benchmark.comparisons[0],
        adaptive_result=_adaptive_result(benchmark),
        trace=trace,
        convergence=convergence,
        recommendation=recommendation,
    )

    assert trace.total_contact_episode_count > 1
    assert convergence.overall_status is not ReferenceConvergenceStatus.CONVERGED
    assert AdaptiveFailureReason.MULTIPLE_CONTACT_EPISODES in {attribution.primary_reason, *attribution.secondary_reasons}
    assert AdaptiveFailureReason.REFERENCE_NOT_CONVERGED in {attribution.primary_reason, *attribution.secondary_reasons}


def test_sphere_sphere_convergence_and_attribution_dataset_runs() -> None:
    case = SphereSphereBenchmarkCase("headon", 1.0 / 240.0, (0.01, 0.3), total_simulation_time=0.5)
    recommendation = SubstepRecommendationConfig(maximum_substeps=8)
    benchmark = run_contact_benchmark((case,), recommendation=recommendation)
    diagnostics = build_attribution_dataset(
        cases=(case,),
        benchmark=benchmark,
        recommendation=recommendation,
    )

    assert diagnostics.traces[0].total_contact_episode_count >= 1
    assert diagnostics.traces[0].first_contact_time is not None
    assert diagnostics.summary.total_cases == 1
    assert diagnostics.summary.convergence_checked_cases == 1
