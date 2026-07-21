from pathlib import Path

from physical_simulation.evaluation import (
    AdaptiveDiagnosticTrace,
    AdaptiveFailureReason,
    AttributionScope,
    BenchmarkComparison,
    BenchmarkValidity,
    ContactEpisodeKind,
    ContactEpisodeMetrics,
    EpisodeMatch,
    EpisodeMatchStatus,
    EpisodeReferenceConvergenceResult,
    ImprovementOutcome,
    PrimaryAttributionDataset,
    PrimaryImpactAttributionInput,
    PrimaryImpactCaseOutcome,
    ReferenceConvergenceStatus,
    ReferenceMetricConvergence,
    attribute_primary_impact_failure,
    build_primary_attribution_markdown_report,
    build_primary_attribution_summary,
    build_run_primary_difference,
    export_primary_attribution_csv,
    export_primary_attribution_json,
    export_run_primary_difference_csv,
    primary_improvement_rates,
    write_primary_attribution_markdown_report,
)


def _episode(index=0, *, restitution=0.4, penetration=0.01, duration=0.01, validity=BenchmarkValidity.VALID):
    return ContactEpisodeMetrics(
        episode_index=index,
        candidate_id="candidate",
        kind=ContactEpisodeKind.PRIMARY_IMPACT,
        body_a_id="a",
        body_b_id="b",
        start_time=0.1,
        end_time=0.1 + duration,
        duration_seconds=duration,
        raw_interval_count=1,
        merged_gap_count=0,
        pre_contact_normal_velocity=-1.0,
        impact_speed=1.0,
        post_contact_normal_velocity=restitution,
        separation_speed=restitution,
        restitution=restitution,
        maximum_penetration=penetration,
        normalized_penetration=penetration / 0.1,
        maximum_penetration_time=0.105,
        contact_sample_count=5,
        physics_step_count=5,
        minimum_timestep=0.001,
        maximum_timestep=0.001,
        minimum_substep_count=16,
        maximum_substep_count=16,
        state_at_start=None,
        state_at_maximum_penetration=None,
        state_at_end=None,
        ended_with_separation=True,
        ended_while_contact_active=False,
        validity=validity,
    )


def _match(status=EpisodeMatchStatus.MATCHED):
    return EpisodeMatch(
        candidate_id="candidate",
        reference_episode_index=0,
        comparison_episode_index=0,
        status=status,
        reference_kind=ContactEpisodeKind.PRIMARY_IMPACT,
        comparison_kind=ContactEpisodeKind.PRIMARY_IMPACT,
        start_time_difference=0.0,
        impact_speed_difference=0.0,
        reason="test",
    )


def _metric(status=ReferenceConvergenceStatus.CONVERGED):
    return ReferenceMetricConvergence("metric", 0.01, 0.0, 0.0, 0.005, 0.02, status)


def _convergence(status=ReferenceConvergenceStatus.CONVERGED):
    return EpisodeReferenceConvergenceResult(
        case_id="case",
        candidate_id="candidate",
        episode_kind=ContactEpisodeKind.PRIMARY_IMPACT,
        levels=(),
        restitution=_metric(status),
        maximum_penetration=_metric(status),
        contact_duration=_metric(status),
        start_time=_metric(status),
        overall_status=status,
    )


def _comparison(coarse_e=0.2, adaptive_e=0.01, coarse_p=0.002, adaptive_p=0.0001, coarse_d=0.002, adaptive_d=0.0001):
    from physical_simulation.evaluation import PrimaryImpactBenchmarkComparison

    return PrimaryImpactBenchmarkComparison(
        case_id="case",
        candidate_id="candidate",
        reference_episode=_episode(),
        coarse_episode=_episode(restitution=0.6, penetration=0.012, duration=0.012),
        adaptive_episode=_episode(restitution=0.41, penetration=0.0101, duration=0.0101),
        coarse_match_status=EpisodeMatchStatus.MATCHED,
        adaptive_match_status=EpisodeMatchStatus.MATCHED,
        coarse_restitution_error=coarse_e,
        adaptive_restitution_error=adaptive_e,
        coarse_penetration_error=coarse_p,
        adaptive_penetration_error=adaptive_p,
        coarse_duration_error=coarse_d,
        adaptive_duration_error=adaptive_d,
        adaptive_improves_restitution=adaptive_e <= coarse_e,
        adaptive_improves_penetration=adaptive_p <= coarse_p,
    )


def _input(**updates):
    data = dict(
        case_id="case",
        candidate_id="candidate",
        coarse_episode=_episode(),
        adaptive_episode=_episode(),
        reference_episode=_episode(),
        coarse_match=_match(),
        adaptive_match=_match(),
        primary_comparison=_comparison(),
        primary_reference_convergence=_convergence(),
        run_level_comparison=BenchmarkComparison(
            case_id="case",
            coarse_restitution_error=0.2,
            adaptive_restitution_error=0.1,
            coarse_penetration_error=0.002,
            adaptive_penetration_error=0.001,
            coarse_rebound_velocity_error=0.2,
            adaptive_rebound_velocity_error=0.1,
            adaptive_step_ratio=0.1,
            adaptive_step_saving=0.9,
            adaptive_improves_restitution=True,
            adaptive_improves_penetration=True,
        ),
        run_level_reference_convergence=None,
        adaptive_trace=None,
    )
    data.update(updates)
    return PrimaryImpactAttributionInput(**data)


def test_primary_scope_improved_has_no_failure_reason() -> None:
    result = attribute_primary_impact_failure(_input())

    assert result.scope is AttributionScope.PRIMARY_IMPACT
    assert result.case_outcome is PrimaryImpactCaseOutcome.IMPROVED
    assert result.primary_reason is AdaptiveFailureReason.NONE
    assert result.restitution_outcome is ImprovementOutcome.IMPROVED


def test_invalid_adaptive_primary_has_highest_priority() -> None:
    result = attribute_primary_impact_failure(
        _input(adaptive_episode=_episode(validity=BenchmarkValidity.NONPHYSICAL_REBOUND))
    )

    assert result.case_outcome is PrimaryImpactCaseOutcome.INVALID_ADAPTIVE
    assert result.primary_reason is AdaptiveFailureReason.NONPHYSICAL_ADAPTIVE_RESULT


def test_primary_unresolved_does_not_fallback_to_run_level() -> None:
    result = attribute_primary_impact_failure(
        _input(primary_reference_convergence=_convergence(ReferenceConvergenceStatus.NOT_CONVERGED))
    )

    assert result.scope is AttributionScope.PRIMARY_IMPACT
    assert result.case_outcome is PrimaryImpactCaseOutcome.REFERENCE_UNRESOLVED
    assert result.primary_reason is AdaptiveFailureReason.REFERENCE_NOT_CONVERGED


def test_episode_mismatch_falls_back_to_run_level_with_secondary_reason() -> None:
    result = attribute_primary_impact_failure(
        _input(adaptive_match=_match(EpisodeMatchStatus.AMBIGUOUS))
    )

    assert result.scope is AttributionScope.FALLBACK_RUN_LEVEL
    assert AdaptiveFailureReason.EPISODE_MISMATCH in result.secondary_reasons


def test_metric_not_improved_gets_specific_primary_reason() -> None:
    result = attribute_primary_impact_failure(
        _input(primary_comparison=_comparison(coarse_e=0.01, adaptive_e=0.03))
    )

    assert result.case_outcome is PrimaryImpactCaseOutcome.PARTIALLY_IMPROVED
    assert result.primary_reason is AdaptiveFailureReason.PRIMARY_RESTITUTION_NOT_IMPROVED


def test_both_acceptable_and_improvement_rates_filter_denominator() -> None:
    acceptable = attribute_primary_impact_failure(
        _input(primary_comparison=_comparison(
            coarse_e=0.001,
            adaptive_e=0.002,
            coarse_p=0.0001,
            adaptive_p=0.0002,
            coarse_d=0.0001,
            adaptive_d=0.0002,
        ))
    )
    unresolved = attribute_primary_impact_failure(
        _input(primary_reference_convergence=_convergence(ReferenceConvergenceStatus.NOT_CONVERGED))
    )
    rates = primary_improvement_rates((acceptable, unresolved))

    assert acceptable.case_outcome is PrimaryImpactCaseOutcome.BOTH_ACCEPTABLE
    assert rates["case"] == 1.0
    assert rates["restitution"] == 1.0


def test_run_primary_difference_secondary_contamination() -> None:
    attr = attribute_primary_impact_failure(_input())
    diff = build_run_primary_difference(
        attr,
        run_level_restitution_outcome=ImprovementOutcome.NOT_IMPROVED,
        run_level_penetration_outcome=ImprovementOutcome.IMPROVED,
        secondary_episode_count=2,
        chatter_count=0,
    )

    assert diff.run_level_difference_explained_by_secondary_episodes is True


def test_summary_and_exports(tmp_path: Path) -> None:
    attr = attribute_primary_impact_failure(_input())
    diff = build_run_primary_difference(
        attr,
        run_level_restitution_outcome=ImprovementOutcome.NOT_IMPROVED,
        run_level_penetration_outcome=ImprovementOutcome.IMPROVED,
        secondary_episode_count=2,
        chatter_count=0,
    )
    summary = build_primary_attribution_summary((attr,), differences=(diff,), run_level_comparisons=(_input().run_level_comparison,))
    dataset = PrimaryAttributionDataset(
        attributions=(attr,),
        differences=(diff,),
        summary=summary,
        config={},
    )

    export_primary_attribution_csv(dataset.attributions, tmp_path / "primary.csv")
    export_run_primary_difference_csv(dataset.differences, tmp_path / "diff.csv")
    export_primary_attribution_json(dataset, tmp_path / "diag.json")
    write_primary_attribution_markdown_report(dataset, tmp_path / "report.md")

    assert summary.primary_scope_cases == 1
    assert "primary_impact" in (tmp_path / "primary.csv").read_text(encoding="utf8")
    assert "explained_by_secondary" in (tmp_path / "diff.csv").read_text(encoding="utf8")
    assert "attributions" in (tmp_path / "diag.json").read_text(encoding="utf8")
    assert "Primary Impact Attribution" in (tmp_path / "report.md").read_text(encoding="utf8")
    assert "Overview" in build_primary_attribution_markdown_report(dataset)
