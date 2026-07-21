from pathlib import Path

from physical_simulation.evaluation import (
    BenchmarkMode,
    BenchmarkValidity,
    ContactEpisodeAnalysisDataset,
    ContactEpisodeKind,
    ContactEpisodeSample,
    ContactEpisodeSegmentationConfig,
    EpisodeMatchStatus,
    EpisodeMatchingConfig,
    extract_raw_contact_intervals,
    match_contact_episodes,
    segment_contact_episodes,
    build_primary_impact_comparison,
    build_contact_episode_statistics,
    export_contact_episodes_csv,
    export_episode_matches_csv,
    export_episode_comparisons_csv,
    export_episode_diagnostics_json,
    write_episode_markdown_report,
    compare_matched_episodes,
)


def _sample(index: int, active: bool, vn: float, *, penetration: float = 0.0, candidate="c"):
    return ContactEpisodeSample(
        simulation_time=index * 0.01,
        physics_step_index=index,
        macro_step_index=None,
        candidate_id=candidate,
        body_a_id="a",
        body_b_id="b",
        active_contact=active,
        gap_or_distance=None,
        penetration=penetration,
        contact_normal=(1.0, 0.0, 0.0),
        normal_relative_velocity=vn,
        body_a_linear_velocity=(0.0, 0.0, 0.0),
        body_b_linear_velocity=(vn, 0.0, 0.0),
        body_a_angular_velocity=(0.0, 0.0, 0.0),
        body_b_angular_velocity=(0.0, 0.0, 0.0),
        substep_count=1,
        timestep=0.01,
        adaptive_state=None,
    )


def _config():
    return ContactEpisodeSegmentationConfig(
        maximum_chatter_gap_seconds=0.021,
        maximum_chatter_gap_steps=2,
        minimum_separation_speed=0.02,
        minimum_impact_speed=0.02,
    )


def test_raw_interval_extraction_and_active_end() -> None:
    samples = (_sample(0, False, -1.0), _sample(1, True, -0.5), _sample(2, True, 0.1))
    intervals = extract_raw_contact_intervals(samples)

    assert len(intervals) == 1
    assert intervals[0].start_sample_index == 1
    assert intervals[0].end_sample_index == 2
    assert intervals[0].ended_while_contact_active is True


def test_short_gap_chatter_merges_into_one_episode() -> None:
    samples = (
        _sample(0, False, -1.0),
        _sample(1, True, -0.5, penetration=0.01),
        _sample(2, False, 0.0),
        _sample(3, True, -0.1, penetration=0.02),
        _sample(4, False, 0.3),
    )
    episodes = segment_contact_episodes(samples, config=_config())

    assert len(episodes) == 1
    assert episodes[0].raw_interval_count == 2
    assert episodes[0].merged_gap_count == 1
    assert episodes[0].restitution == 0.3
    assert episodes[0].maximum_penetration_time == 0.03


def test_clear_separation_does_not_merge_and_secondary_is_classified() -> None:
    samples = (
        _sample(0, False, -1.0),
        _sample(1, True, -0.5, penetration=0.01),
        _sample(2, False, 0.1),
        _sample(3, False, 0.1),
        _sample(4, True, -0.4, penetration=0.02),
        _sample(5, False, 0.2),
    )
    episodes = segment_contact_episodes(samples, config=_config())

    assert [episode.kind for episode in episodes] == [
        ContactEpisodeKind.PRIMARY_IMPACT,
        ContactEpisodeKind.SECONDARY_IMPACT,
    ]
    assert episodes[0].restitution == 0.1
    assert episodes[1].restitution == 0.5


def test_missing_separation_keeps_restitution_none() -> None:
    samples = (_sample(0, False, -1.0), _sample(1, True, -0.5, penetration=0.01))
    episode = segment_contact_episodes(samples, config=_config())[0]

    assert episode.ended_while_contact_active is True
    assert episode.restitution is None
    assert episode.validity is BenchmarkValidity.TIMEOUT


def test_episode_matching_matched_unmatched_and_ambiguous() -> None:
    reference = segment_contact_episodes(
        (_sample(0, False, -1.0), _sample(1, True, -0.5, penetration=0.01), _sample(2, False, 0.2)),
        config=_config(),
    )
    comparison = segment_contact_episodes(
        (_sample(0, False, -1.0), _sample(1, True, -0.5, penetration=0.011), _sample(2, False, 0.2)),
        config=_config(),
    )
    matches = match_contact_episodes(
        reference=reference,
        comparison=comparison,
        config=EpisodeMatchingConfig(maximum_start_time_difference=0.02),
    )

    assert matches[0].status is EpisodeMatchStatus.MATCHED
    assert match_contact_episodes(reference=reference, comparison=(), config=EpisodeMatchingConfig(0.02))[0].status is EpisodeMatchStatus.UNMATCHED_REFERENCE

    ambiguous = match_contact_episodes(
        reference=reference,
        comparison=(comparison[0], comparison[0]),
        config=EpisodeMatchingConfig(maximum_start_time_difference=0.02),
    )
    assert ambiguous[0].status is EpisodeMatchStatus.AMBIGUOUS


def test_primary_comparison_and_exports(tmp_path: Path) -> None:
    ref = segment_contact_episodes(
        (_sample(0, False, -1.0), _sample(1, True, -0.5, penetration=0.01), _sample(2, False, 0.2)),
        config=_config(),
    )
    comp = segment_contact_episodes(
        (_sample(0, False, -1.0), _sample(1, True, -0.5, penetration=0.02), _sample(2, False, 0.1)),
        config=_config(),
    )
    matches = match_contact_episodes(reference=ref, comparison=comp, config=EpisodeMatchingConfig(0.02))
    comparisons = compare_matched_episodes(
        case_id="case",
        reference_mode=BenchmarkMode.FIXED_FINE,
        comparison_mode=BenchmarkMode.ADAPTIVE,
        reference=ref,
        comparison=comp,
        matches=matches,
    )
    primary = build_primary_impact_comparison(
        case_id="case",
        reference=ref,
        coarse=comp,
        adaptive=comp,
        matching=EpisodeMatchingConfig(0.02),
    )
    stats = build_contact_episode_statistics(ref, extract_raw_contact_intervals((
        _sample(0, False, -1.0), _sample(1, True, -0.5), _sample(2, False, 0.2)
    )))
    dataset = ContactEpisodeAnalysisDataset(
        case_id="case",
        episodes_by_mode={"fixed_fine": ref, "adaptive": comp},
        matches=matches,
        comparisons=comparisons,
        primary_comparison=primary,
        reference_convergence=None,
        statistics_by_mode={"fixed_fine": stats},
    )

    export_contact_episodes_csv((("case", BenchmarkMode.FIXED_FINE, ref[0]),), tmp_path / "episodes.csv")
    export_episode_matches_csv(matches, tmp_path / "matches.csv")
    export_episode_comparisons_csv(comparisons, tmp_path / "comparisons.csv")
    export_episode_diagnostics_json(dataset, tmp_path / "diagnostics.json")
    write_episode_markdown_report(dataset, tmp_path / "report.md")

    assert primary is not None
    assert comparisons[0].penetration_error == 0.01
    assert "primary_impact" in (tmp_path / "episodes.csv").read_text(encoding="utf8")
    assert "matched" in (tmp_path / "matches.csv").read_text(encoding="utf8")
    assert "penetration_error" in (tmp_path / "comparisons.csv").read_text(encoding="utf8")
    assert "episodes_by_mode" in (tmp_path / "diagnostics.json").read_text(encoding="utf8")
    assert "Primary Impact Comparison" in (tmp_path / "report.md").read_text(encoding="utf8")
