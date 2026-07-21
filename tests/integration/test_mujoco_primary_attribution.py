from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.evaluation import (
    AdaptiveFailureReason,
    AttributionScope,
    BenchmarkMode,
    ContactEpisodeSegmentationConfig,
    EpisodeMatchStatus,
    EpisodeMatchingConfig,
    ImprovementOutcome,
    PrimaryImpactAttributionInput,
    PrimaryImpactCaseOutcome,
    ReferenceConvergenceConfig,
    ReferenceConvergenceResult,
    ReferenceConvergenceStatus,
    SpherePlaneBenchmarkCase,
    SphereSphereBenchmarkCase,
    attribute_primary_impact_failure,
    build_primary_impact_comparison,
    build_run_primary_difference,
    collect_contact_episode_samples,
    run_adaptive_diagnostic_trace,
    run_contact_benchmark,
    run_episode_reference_convergence,
    run_reference_convergence,
    segment_contact_episodes,
)
from physical_simulation.mujoco import AdaptiveSubstepConfig, SubstepRecommendationConfig


def _primary_input(case, *, recommendation=None, adaptive_config=None, run_reference=None):
    rec = recommendation or SubstepRecommendationConfig(maximum_substeps=16)
    seg = ContactEpisodeSegmentationConfig(maximum_chatter_gap_seconds=2.0 * case.macro_timestep / rec.maximum_substeps)
    matching = EpisodeMatchingConfig(maximum_start_time_difference=case.macro_timestep)
    benchmark = run_contact_benchmark((case,), recommendation=rec)
    episodes = {
        mode: segment_contact_episodes(
            collect_contact_episode_samples(case, mode=mode, recommendation=rec),
            config=seg,
        )
        for mode in (BenchmarkMode.FIXED_COARSE, BenchmarkMode.FIXED_FINE, BenchmarkMode.ADAPTIVE)
    }
    primary = build_primary_impact_comparison(
        case_id=case.case_id,
        reference=episodes[BenchmarkMode.FIXED_FINE],
        coarse=episodes[BenchmarkMode.FIXED_COARSE],
        adaptive=episodes[BenchmarkMode.ADAPTIVE],
        matching=matching,
    )
    convergence = run_episode_reference_convergence(case, segmentation=seg, recommendation=rec, matching_config=matching)
    trace = run_adaptive_diagnostic_trace(case, recommendation=rec, adaptive_config=adaptive_config)
    run_ref = run_reference if run_reference is not None else run_reference_convergence(case, recommendation=rec)
    return PrimaryImpactAttributionInput(
        case_id=case.case_id,
        candidate_id=primary.candidate_id if primary is not None else f"{case.case_id}_candidate",
        coarse_episode=None if primary is None else primary.coarse_episode,
        adaptive_episode=None if primary is None else primary.adaptive_episode,
        reference_episode=None if primary is None else primary.reference_episode,
        coarse_match=None,
        adaptive_match=None if primary is None else type("M", (), {"status": primary.adaptive_match_status})(),
        primary_comparison=primary,
        primary_reference_convergence=convergence,
        run_level_comparison=benchmark.comparisons[0],
        run_level_reference_convergence=run_ref,
        adaptive_trace=trace,
    ), benchmark, episodes


def test_primary_and_run_level_improved_has_no_failure_reason() -> None:
    case = SpherePlaneBenchmarkCase("good", 1.0, 1.0 / 240.0, (0.02, 0.3), total_simulation_time=0.6)
    input, _, _ = _primary_input(case)
    attribution = attribute_primary_impact_failure(input)

    assert attribution.scope is AttributionScope.PRIMARY_IMPACT
    assert attribution.case_outcome is PrimaryImpactCaseOutcome.IMPROVED
    assert attribution.primary_reason is AdaptiveFailureReason.NONE


def test_run_level_unresolved_does_not_override_converged_primary() -> None:
    case = SpherePlaneBenchmarkCase("run_unresolved", 1.0, 1.0 / 240.0, (0.02, 0.3), total_simulation_time=0.6)
    unresolved = ReferenceConvergenceResult(
        case_id=case.case_id,
        levels=(),
        restitution=None,
        rebound_speed=None,
        maximum_penetration=None,
        contact_duration=None,
        overall_status=ReferenceConvergenceStatus.NOT_CONVERGED,
        selected_reference_level=None,
    )
    input, _, _ = _primary_input(case, run_reference=unresolved)
    attribution = attribute_primary_impact_failure(input)

    assert attribution.primary_reference_status is ReferenceConvergenceStatus.CONVERGED
    assert attribution.run_level_reference_status is ReferenceConvergenceStatus.NOT_CONVERGED
    assert attribution.primary_reason is not AdaptiveFailureReason.REFERENCE_NOT_CONVERGED


def test_episode_mismatch_falls_back_to_run_level() -> None:
    case = SpherePlaneBenchmarkCase("mismatch", 1.0, 1.0 / 240.0, (0.02, 0.3), total_simulation_time=0.6)
    input, _, _ = _primary_input(case)
    bad_input = PrimaryImpactAttributionInput(
        **{
            **input.__dict__,
            "adaptive_match": type("M", (), {"status": EpisodeMatchStatus.AMBIGUOUS})(),
        }
    )
    attribution = attribute_primary_impact_failure(bad_input)

    assert attribution.scope is AttributionScope.FALLBACK_RUN_LEVEL
    assert AdaptiveFailureReason.EPISODE_MISMATCH in attribution.secondary_reasons


def test_short_prediction_only_failure_when_primary_metric_not_improved() -> None:
    case = SpherePlaneBenchmarkCase("short", 1.0, 1.0 / 240.0, (0.02, 0.3), total_simulation_time=0.6)
    rec = SubstepRecommendationConfig(maximum_substeps=16)
    adaptive_config = AdaptiveSubstepConfig(
        macro_timestep=case.macro_timestep,
        prediction_horizon_multiplier=0.1,
        recommendation=rec,
    )
    input, _, _ = _primary_input(case, recommendation=rec, adaptive_config=adaptive_config)
    attribution = attribute_primary_impact_failure(input)

    assert attribution.primary_reason is AdaptiveFailureReason.NONE
    assert any("prediction lead" in item for item in attribution.evidence)


def test_max_substeps_limited_can_be_secondary_when_primary_improves() -> None:
    case = SpherePlaneBenchmarkCase("limited", 1.0, 1.0 / 240.0, (0.005, 0.3), total_simulation_time=1.0)
    rec = SubstepRecommendationConfig(maximum_substeps=2)
    input, _, _ = _primary_input(case, recommendation=rec)
    attribution = attribute_primary_impact_failure(input)

    assert attribution.scope is AttributionScope.PRIMARY_IMPACT
    if attribution.primary_reason is AdaptiveFailureReason.NONE:
        assert AdaptiveFailureReason.MAX_SUBSTEPS_LIMITED not in attribution.secondary_reasons or attribution.case_outcome is not PrimaryImpactCaseOutcome.IMPROVED


def test_secondary_episode_difference_can_explain_run_level_difference() -> None:
    case = SpherePlaneBenchmarkCase("secondary", 1.0, 1.0 / 240.0, (0.005, 0.3), total_simulation_time=1.0)
    input, _, _ = _primary_input(case)
    attribution = attribute_primary_impact_failure(input)
    difference = build_run_primary_difference(
        attribution,
        run_level_restitution_outcome=ImprovementOutcome.NOT_IMPROVED,
        run_level_penetration_outcome=ImprovementOutcome.IMPROVED,
        secondary_episode_count=2,
        chatter_count=0,
    )

    assert difference.run_level_difference_explained_by_secondary_episodes is True


def test_sphere_sphere_primary_attribution_runs() -> None:
    case = SphereSphereBenchmarkCase("headon", 1.0 / 240.0, (0.01, 0.3), total_simulation_time=0.5)
    input, _, _ = _primary_input(case, recommendation=SubstepRecommendationConfig(maximum_substeps=8))
    attribution = attribute_primary_impact_failure(input)

    assert attribution.candidate_id == "headon_candidate"
    assert attribution.scope in {AttributionScope.PRIMARY_IMPACT, AttributionScope.FALLBACK_RUN_LEVEL}
