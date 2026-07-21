from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.evaluation import (
    BenchmarkMode,
    ContactEpisodeKind,
    ContactEpisodeSegmentationConfig,
    EpisodeMatchStatus,
    EpisodeMatchingConfig,
    ReferenceConvergenceStatus,
    SpherePlaneBenchmarkCase,
    SphereSphereBenchmarkCase,
    build_primary_impact_comparison,
    collect_contact_episode_samples,
    match_contact_episodes,
    run_episode_reference_convergence,
    segment_contact_episodes,
)
from physical_simulation.mujoco import SubstepRecommendationConfig


def _seg(case, rec):
    return ContactEpisodeSegmentationConfig(maximum_chatter_gap_seconds=2.0 * case.macro_timestep / rec.maximum_substeps)


def test_single_sphere_plane_primary_impact_matches_modes() -> None:
    case = SpherePlaneBenchmarkCase("single", 1.0, 1.0 / 240.0, (0.02, 0.3), total_simulation_time=0.6)
    rec = SubstepRecommendationConfig(maximum_substeps=16)
    seg = _seg(case, rec)
    fine = segment_contact_episodes(collect_contact_episode_samples(case, mode=BenchmarkMode.FIXED_FINE, recommendation=rec), config=seg)
    coarse = segment_contact_episodes(collect_contact_episode_samples(case, mode=BenchmarkMode.FIXED_COARSE, recommendation=rec), config=seg)
    adaptive = segment_contact_episodes(collect_contact_episode_samples(case, mode=BenchmarkMode.ADAPTIVE, recommendation=rec), config=seg)
    primary = build_primary_impact_comparison(
        case_id=case.case_id,
        reference=fine,
        coarse=coarse,
        adaptive=adaptive,
        matching=EpisodeMatchingConfig(maximum_start_time_difference=case.macro_timestep),
    )

    assert [episode.kind for episode in fine] == [ContactEpisodeKind.PRIMARY_IMPACT]
    assert primary is not None
    assert primary.coarse_match_status is EpisodeMatchStatus.MATCHED
    assert primary.adaptive_match_status is EpisodeMatchStatus.MATCHED
    assert primary.reference_episode.restitution is not None


def test_multiple_bounces_produce_secondary_impacts() -> None:
    case = SpherePlaneBenchmarkCase("multi", 0.4, 1.0 / 240.0, (0.02, 0.5), total_simulation_time=0.7)
    rec = SubstepRecommendationConfig(maximum_substeps=8)
    episodes = segment_contact_episodes(
        collect_contact_episode_samples(case, mode=BenchmarkMode.ADAPTIVE, recommendation=rec),
        config=_seg(case, rec),
    )

    assert any(episode.kind is ContactEpisodeKind.PRIMARY_IMPACT for episode in episodes)
    assert any(episode.kind is ContactEpisodeKind.SECONDARY_IMPACT for episode in episodes)


def test_resting_contact_has_no_restitution_when_contact_stays_active() -> None:
    case = SpherePlaneBenchmarkCase("resting", 0.1, 1.0 / 240.0, (0.02, 1.0), total_simulation_time=0.4)
    rec = SubstepRecommendationConfig(maximum_substeps=8)
    episodes = segment_contact_episodes(
        collect_contact_episode_samples(case, mode=BenchmarkMode.FIXED_FINE, recommendation=rec),
        config=_seg(case, rec),
    )

    assert episodes
    assert episodes[-1].ended_while_contact_active is True
    assert episodes[-1].restitution is None


def test_sphere_sphere_primary_episode_and_match() -> None:
    case = SphereSphereBenchmarkCase("headon", 1.0 / 240.0, (0.01, 0.3), total_simulation_time=0.5)
    rec = SubstepRecommendationConfig(maximum_substeps=8)
    seg = _seg(case, rec)
    fine = segment_contact_episodes(collect_contact_episode_samples(case, mode=BenchmarkMode.FIXED_FINE, recommendation=rec), config=seg)
    adaptive = segment_contact_episodes(collect_contact_episode_samples(case, mode=BenchmarkMode.ADAPTIVE, recommendation=rec), config=seg)
    matches = match_contact_episodes(
        reference=fine,
        comparison=adaptive,
        config=EpisodeMatchingConfig(maximum_start_time_difference=case.macro_timestep),
    )

    assert fine[0].kind is ContactEpisodeKind.PRIMARY_IMPACT
    assert matches[0].status is EpisodeMatchStatus.MATCHED


def test_episode_reference_convergence_can_converge_primary_when_run_has_more_events() -> None:
    case = SpherePlaneBenchmarkCase("converge", 1.0, 1.0 / 240.0, (0.02, 0.3), total_simulation_time=0.6)
    rec = SubstepRecommendationConfig(maximum_substeps=16)
    result = run_episode_reference_convergence(case, segmentation=_seg(case, rec), recommendation=rec)

    assert result.overall_status is ReferenceConvergenceStatus.CONVERGED
    assert result.episode_kind is ContactEpisodeKind.PRIMARY_IMPACT
