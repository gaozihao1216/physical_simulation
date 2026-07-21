"""Analyze contact episodes and primary-impact matching across benchmark modes."""

from __future__ import annotations

from pathlib import Path

from physical_simulation.evaluation import (
    BenchmarkMode,
    ContactEpisodeAnalysisDataset,
    ContactEpisodeKind,
    ContactEpisodeSegmentationConfig,
    EpisodeMatchingConfig,
    SpherePlaneBenchmarkCase,
    build_contact_episode_statistics,
    build_primary_impact_comparison,
    collect_contact_episode_samples,
    compare_matched_episodes,
    export_contact_episodes_csv,
    export_episode_comparisons_csv,
    export_episode_diagnostics_json,
    export_episode_matches_csv,
    extract_raw_contact_intervals,
    match_contact_episodes,
    run_episode_reference_convergence,
    segment_contact_episodes,
    write_episode_markdown_report,
)
from physical_simulation.mujoco import SubstepRecommendationConfig


def main() -> None:
    case = SpherePlaneBenchmarkCase("episode_drop_h1_dt240_solref0.02_0.3", 1.0, 1.0 / 240.0, (0.02, 0.3), total_simulation_time=0.6)
    recommendation = SubstepRecommendationConfig(maximum_substeps=16)
    segmentation = ContactEpisodeSegmentationConfig(
        maximum_chatter_gap_seconds=2.0 * case.macro_timestep / recommendation.maximum_substeps
    )
    matching = EpisodeMatchingConfig(maximum_start_time_difference=case.macro_timestep)

    episodes_by_mode = {}
    raw_by_mode = {}
    for mode in (BenchmarkMode.FIXED_COARSE, BenchmarkMode.FIXED_FINE, BenchmarkMode.ADAPTIVE):
        samples = collect_contact_episode_samples(case, mode=mode, recommendation=recommendation)
        raw = extract_raw_contact_intervals(samples)
        episodes = segment_contact_episodes(samples, config=segmentation)
        raw_by_mode[mode.value] = raw
        episodes_by_mode[mode.value] = episodes

    reference = episodes_by_mode[BenchmarkMode.FIXED_FINE.value]
    coarse = episodes_by_mode[BenchmarkMode.FIXED_COARSE.value]
    adaptive = episodes_by_mode[BenchmarkMode.ADAPTIVE.value]
    coarse_matches = match_contact_episodes(reference=reference, comparison=coarse, config=matching)
    adaptive_matches = match_contact_episodes(reference=reference, comparison=adaptive, config=matching)
    comparisons = (
        *compare_matched_episodes(
            case_id=case.case_id,
            reference_mode=BenchmarkMode.FIXED_FINE,
            comparison_mode=BenchmarkMode.FIXED_COARSE,
            reference=reference,
            comparison=coarse,
            matches=coarse_matches,
        ),
        *compare_matched_episodes(
            case_id=case.case_id,
            reference_mode=BenchmarkMode.FIXED_FINE,
            comparison_mode=BenchmarkMode.ADAPTIVE,
            reference=reference,
            comparison=adaptive,
            matches=adaptive_matches,
        ),
    )
    primary = build_primary_impact_comparison(
        case_id=case.case_id,
        reference=reference,
        coarse=coarse,
        adaptive=adaptive,
        matching=matching,
    )
    convergence = run_episode_reference_convergence(
        case,
        segmentation=segmentation,
        recommendation=recommendation,
        matching_config=matching,
    )
    stats = {
        mode: build_contact_episode_statistics(episodes_by_mode[mode], raw_by_mode[mode])
        for mode in episodes_by_mode
    }
    dataset = ContactEpisodeAnalysisDataset(
        case_id=case.case_id,
        episodes_by_mode=episodes_by_mode,
        matches=(*coarse_matches, *adaptive_matches),
        comparisons=comparisons,
        primary_comparison=primary,
        reference_convergence=convergence,
        statistics_by_mode=stats,
    )

    output_dir = Path("artifacts/contact_episodes")
    rows = [
        (case.case_id, BenchmarkMode(mode), episode)
        for mode, episodes in episodes_by_mode.items()
        for episode in episodes
    ]
    export_contact_episodes_csv(rows, output_dir / "episodes.csv")
    export_episode_matches_csv(dataset.matches, output_dir / "matches.csv")
    export_episode_comparisons_csv(dataset.comparisons, output_dir / "comparisons.csv")
    export_episode_diagnostics_json(dataset, output_dir / "diagnostics.json")
    write_episode_markdown_report(dataset, output_dir / "report.md")

    print(f"raw interval count: {sum(len(raw) for raw in raw_by_mode.values())}")
    print(f"merged episode count: {sum(len(episodes) for episodes in episodes_by_mode.values())}")
    for mode, episodes in episodes_by_mode.items():
        print(
            f"{mode}: primary={_count(episodes, ContactEpisodeKind.PRIMARY_IMPACT)}, "
            f"secondary={_count(episodes, ContactEpisodeKind.SECONDARY_IMPACT)}, "
            f"resting={_count(episodes, ContactEpisodeKind.RESTING_CONTACT)}, "
            f"chatter={_count(episodes, ContactEpisodeKind.CONTACT_CHATTER)}"
        )
    if primary is not None:
        ref = primary.reference_episode
        print(
            "Episode 0 reference: "
            f"start={ref.start_time:.6f}, impact={_fmt(ref.impact_speed)}, "
            f"separation={_fmt(ref.separation_speed)}, e={_fmt(ref.restitution)}, "
            f"penetration={ref.maximum_penetration:.6f}, duration={ref.duration_seconds:.6f}"
        )
        print(f"coarse match status: {primary.coarse_match_status.value}")
        print(f"adaptive match status: {primary.adaptive_match_status.value}")
    print(f"primary reference convergence status: {convergence.overall_status.value}")
    print(f"exports: {output_dir / 'episodes.csv'}, {output_dir / 'matches.csv'}, {output_dir / 'comparisons.csv'}, {output_dir / 'diagnostics.json'}, {output_dir / 'report.md'}")


def _count(episodes, kind: ContactEpisodeKind) -> int:
    return sum(episode.kind is kind for episode in episodes)


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
