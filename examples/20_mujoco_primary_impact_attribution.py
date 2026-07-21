"""Preferred primary-impact attribution compared with run-level attribution."""

from __future__ import annotations

from pathlib import Path

from physical_simulation.evaluation import (
    AttributionScope,
    BenchmarkMode,
    ContactEpisodeSegmentationConfig,
    EpisodeMatchingConfig,
    ImprovementOutcome,
    PrimaryAttributionDataset,
    PrimaryImpactAttributionInput,
    SpherePlaneBenchmarkCase,
    SphereSphereBenchmarkCase,
    attribute_primary_impact_failure,
    build_primary_attribution_summary,
    build_primary_impact_comparison,
    build_run_primary_difference,
    collect_contact_episode_samples,
    export_primary_attribution_csv,
    export_primary_attribution_json,
    export_run_primary_difference_csv,
    match_contact_episodes,
    primary_improvement_rates,
    run_adaptive_diagnostic_trace,
    run_contact_benchmark,
    run_episode_reference_convergence,
    run_reference_convergence,
    segment_contact_episodes,
    write_primary_attribution_markdown_report,
)
from physical_simulation.mujoco import SubstepRecommendationConfig


def main() -> None:
    cases = (
        SpherePlaneBenchmarkCase("drop_h1_dt240_solref0.02_0.3", 1.0, 1.0 / 240.0, (0.02, 0.3), total_simulation_time=0.6),
        SpherePlaneBenchmarkCase("drop_h1_dt240_solref0.005_0.3", 1.0, 1.0 / 240.0, (0.005, 0.3), total_simulation_time=1.0),
        SphereSphereBenchmarkCase("sphere_sphere_headon_dt240_solref0.01_0.3", 1.0 / 240.0, (0.01, 0.3), total_simulation_time=0.5),
    )
    recommendation = SubstepRecommendationConfig(maximum_substeps=16)
    attributions = []
    differences = []
    run_comparisons = []
    concrete_case = None
    for case in cases:
        attribution, difference, comparison = _run_case(case, recommendation)
        attributions.append(attribution)
        differences.append(difference)
        run_comparisons.append(comparison)
        if concrete_case is None:
            concrete_case = (attribution, difference)

    summary = build_primary_attribution_summary(
        attributions,
        differences=differences,
        run_level_comparisons=run_comparisons,
    )
    dataset = PrimaryAttributionDataset(
        attributions=tuple(attributions),
        differences=tuple(differences),
        summary=summary,
        config={"recommendation": {"maximum_substeps": recommendation.maximum_substeps}},
    )
    output_dir = Path("artifacts/contact_primary_attribution")
    export_primary_attribution_csv(dataset.attributions, output_dir / "primary_attribution.csv")
    export_run_primary_difference_csv(dataset.differences, output_dir / "run_primary_differences.csv")
    export_primary_attribution_json(dataset, output_dir / "diagnostics.json")
    write_primary_attribution_markdown_report(dataset, output_dir / "report.md")

    rates = primary_improvement_rates(dataset.attributions)
    print(f"case count: {summary.total_cases}")
    print(f"primary matched count: {summary.matched_primary_cases}")
    print(f"primary reference converged count: {summary.primary_reference_converged_cases}")
    print(f"fallback run-level count: {summary.fallback_run_level_cases}")
    print(f"primary case outcome counts: {dict(summary.case_outcome_counts)}")
    print(f"primary failure reason counts: {dict(summary.primary_reason_counts)}")
    print(f"run-level unresolved but primary converged count: {summary.run_level_unresolved_but_primary_converged_cases}")
    print(
        "run-level not improved but primary improved count: "
        f"{sum(item.run_level_difference_explained_by_secondary_episodes for item in dataset.differences)}"
    )
    print(f"primary restitution improvement rate: {rates['restitution']:.6f}")
    print(f"primary penetration improvement rate: {rates['penetration']:.6f}")
    print(f"primary duration improvement rate: {rates['duration']:.6f}")
    print(f"mean adaptive step saving: {summary.mean_adaptive_step_saving:.6f}")
    if concrete_case is not None:
        attribution, difference = concrete_case
        print(
            f"example case {attribution.case_id}: scope={attribution.scope.value}, "
            f"primary outcome={attribution.case_outcome.value}, reason={attribution.primary_reason.value}, "
            f"secondary episodes={difference.secondary_episode_count}"
        )
    print(
        "exports: "
        f"{output_dir / 'primary_attribution.csv'}, "
        f"{output_dir / 'run_primary_differences.csv'}, "
        f"{output_dir / 'diagnostics.json'}, "
        f"{output_dir / 'report.md'}"
    )


def _run_case(case, recommendation):
    segmentation = ContactEpisodeSegmentationConfig(
        maximum_chatter_gap_seconds=2.0 * case.macro_timestep / recommendation.maximum_substeps
    )
    matching = EpisodeMatchingConfig(maximum_start_time_difference=case.macro_timestep)
    benchmark = run_contact_benchmark((case,), recommendation=recommendation)
    episodes = {
        mode: segment_contact_episodes(
            collect_contact_episode_samples(case, mode=mode, recommendation=recommendation),
            config=segmentation,
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
    reference = episodes[BenchmarkMode.FIXED_FINE]
    adaptive = episodes[BenchmarkMode.ADAPTIVE]
    coarse = episodes[BenchmarkMode.FIXED_COARSE]
    adaptive_match = match_contact_episodes(reference=reference[:1], comparison=adaptive, config=matching)[0] if reference else None
    coarse_match = match_contact_episodes(reference=reference[:1], comparison=coarse, config=matching)[0] if reference else None
    primary_convergence = run_episode_reference_convergence(
        case,
        segmentation=segmentation,
        recommendation=recommendation,
        matching_config=matching,
    )
    run_convergence = run_reference_convergence(case, recommendation=recommendation)
    trace = run_adaptive_diagnostic_trace(case, recommendation=recommendation)
    input = PrimaryImpactAttributionInput(
        case_id=case.case_id,
        candidate_id=primary.candidate_id if primary is not None else f"{case.case_id}_candidate",
        coarse_episode=None if primary is None else primary.coarse_episode,
        adaptive_episode=None if primary is None else primary.adaptive_episode,
        reference_episode=None if primary is None else primary.reference_episode,
        coarse_match=coarse_match,
        adaptive_match=adaptive_match,
        primary_comparison=primary,
        primary_reference_convergence=primary_convergence,
        run_level_comparison=benchmark.comparisons[0],
        run_level_reference_convergence=run_convergence,
        adaptive_trace=trace,
    )
    attribution = attribute_primary_impact_failure(input)
    run_restitution = (
        ImprovementOutcome.IMPROVED
        if benchmark.comparisons[0].adaptive_improves_restitution
        else ImprovementOutcome.NOT_IMPROVED
    )
    run_penetration = (
        ImprovementOutcome.IMPROVED
        if benchmark.comparisons[0].adaptive_improves_penetration
        else ImprovementOutcome.NOT_IMPROVED
    )
    difference = build_run_primary_difference(
        attribution,
        run_level_restitution_outcome=run_restitution,
        run_level_penetration_outcome=run_penetration,
        secondary_episode_count=max(0, len(adaptive) - 1),
        chatter_count=sum(episode.kind.value == "contact_chatter" for episode in adaptive),
    )
    return attribution, difference, benchmark.comparisons[0]


if __name__ == "__main__":
    main()
