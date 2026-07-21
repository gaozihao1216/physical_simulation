from pathlib import Path

import pytest

from physical_simulation.evaluation import (
    AdaptiveContactEpisodeTrace,
    AdaptiveDiagnosticTrace,
    AdaptiveFailureAttribution,
    AdaptiveFailureReason,
    AttributionBenchmarkSummary,
    AttributionDiagnosticDataset,
    BenchmarkComparison,
    BenchmarkMode,
    BenchmarkValidationConfig,
    BenchmarkValidity,
    ContactBenchmarkDataset,
    ContactBenchmarkResult,
    ImprovementOutcome,
    ReferenceConvergenceConfig,
    ReferenceConvergenceStatus,
    ReferenceLevelResult,
    RestitutionOutcome,
    SpherePlaneBenchmarkCase,
    attribute_adaptive_failure,
    build_attribution_markdown_report,
    build_reference_convergence_result,
    export_attribution_csv,
    export_convergence_csv,
    export_diagnostic_json,
    select_convergence_cases,
    write_attribution_markdown_report,
)
from physical_simulation.validation.errors import PhysicsValidationError
from physical_simulation.mujoco import ContactMotionState


def _level(factor: int, restitution: float | None, penetration: float, validity=BenchmarkValidity.VALID):
    return ReferenceLevelResult(
        refinement_level=factor,
        timestep=1.0 / (240.0 * factor),
        refinement_factor=factor,
        restitution=restitution,
        rebound_speed=None if restitution is None else restitution,
        maximum_penetration=penetration,
        contact_duration_seconds=0.01,
        validity=validity,
        physics_step_count=240 * factor,
    )


def _result(mode: BenchmarkMode, *, validity=BenchmarkValidity.VALID, restitution=0.5, penetration=0.01, steps=100):
    return ContactBenchmarkResult(
        case_id="case",
        mode=mode,
        validity=validity,
        timestep=1.0 / 240.0,
        macro_timestep=1.0 / 240.0,
        total_simulation_time=1.0,
        outcome=RestitutionOutcome.REBOUNDED,
        impact_speed=1.0,
        rebound_speed=restitution,
        restitution=restitution,
        maximum_penetration=penetration,
        normalized_penetration=penetration / 0.1,
        contact_duration_seconds=0.01,
        final_position=(0.0, 0.0, 0.1),
        final_linear_velocity=(0.0, 0.0, 0.0),
        final_angular_velocity=(0.0, 0.0, 0.0),
        macro_step_count=240,
        physics_step_count=steps,
        wall_time_seconds=0.0,
        adaptive_substepped_macro_steps=None,
        adaptive_max_substep_count=None,
    )


def _comparison(case_id="case", coarse_e=0.1, adaptive_e=0.2, coarse_p=0.001, adaptive_p=0.002):
    return BenchmarkComparison(
        case_id=case_id,
        coarse_restitution_error=coarse_e,
        adaptive_restitution_error=adaptive_e,
        coarse_penetration_error=coarse_p,
        adaptive_penetration_error=adaptive_p,
        coarse_rebound_velocity_error=coarse_e,
        adaptive_rebound_velocity_error=adaptive_e,
        adaptive_step_ratio=0.1,
        adaptive_step_saving=0.9,
        adaptive_improves_restitution=adaptive_e <= coarse_e,
        adaptive_improves_penetration=adaptive_p <= coarse_p,
    )


def _trace(*, lead=1.0, limited=False, episodes=1, max_substeps=8):
    episode = AdaptiveContactEpisodeTrace(
        episode_index=0,
        candidate_id="candidate",
        prediction_time=0.1 if lead is not None else None,
        predicted_absolute_contact_time=0.1 if lead is not None else None,
        first_actual_contact_time=0.1 + lead / 240.0 if lead is not None else 0.2,
        prediction_lead_time_seconds=None if lead is None else lead / 240.0,
        prediction_lead_time_macro_steps=lead,
        approaching_start_time=0.1,
        impacting_start_time=0.2,
        separating_start_time=0.3,
        resting_start_time=None,
        episode_end_time=0.4,
        gap_at_first_prediction=0.01,
        approach_speed_at_first_prediction=1.0,
        solver_characteristic_timescale=0.01,
        requested_substep_count=max_substeps,
        maximum_actual_substep_count=max_substeps,
        limited_by_maximum_substeps=limited,
        minimum_actual_timestep=1.0 / (240.0 * max_substeps),
        substepped_macro_step_count=3,
        contact_substep_count=5,
        maximum_penetration=0.01,
        maximum_penetration_time=0.2,
        state_at_maximum_penetration=None,
        contact_duration_seconds=0.02,
    )
    return AdaptiveDiagnosticTrace(
        case_id="case",
        episodes=tuple(episode for _ in range(episodes)),
        total_contact_episode_count=episodes,
        total_substepped_macro_steps=3 * episodes,
        total_physics_steps=100,
        first_prediction_time=0.1 if lead is not None else None,
        first_contact_time=0.2,
        global_minimum_timestep=1.0 / (240.0 * max_substeps),
        global_maximum_substep_count=max_substeps,
    )


def test_reference_convergence_difference_ratio_and_zero_difference() -> None:
    converged = build_reference_convergence_result(
        "case",
        (_level(1, 0.5, 0.010), _level(2, 0.51, 0.011), _level(4, 0.510, 0.011)),
    )

    assert converged.overall_status is ReferenceConvergenceStatus.CONVERGED
    assert converged.restitution.difference_ratio == pytest.approx(0.0)
    assert converged.selected_reference_level == 4


def test_reference_not_converged_and_invalid_status() -> None:
    not_converged = build_reference_convergence_result(
        "case",
        (_level(1, 0.1, 0.01), _level(2, 0.2, 0.02), _level(4, 0.4, 0.05)),
    )
    invalid = build_reference_convergence_result(
        "case",
        (_level(1, 0.1, 0.01), _level(2, None, float("nan"), BenchmarkValidity.UNSTABLE), _level(4, 0.4, 0.05)),
    )

    assert not_converged.overall_status is ReferenceConvergenceStatus.NOT_CONVERGED
    assert invalid.overall_status is ReferenceConvergenceStatus.INVALID_RESULT


def test_reference_config_validation() -> None:
    with pytest.raises(PhysicsValidationError):
        ReferenceConvergenceConfig(refinement_factors=(1, 1))
    with pytest.raises(PhysicsValidationError):
        ReferenceConvergenceConfig(restitution_absolute_tolerance=float("nan"))


def test_selection_top_k_and_dedup_is_deterministic() -> None:
    cases = (
        SpherePlaneBenchmarkCase("a", 0.4, 1.0 / 240.0, (0.02, 0.3)),
        SpherePlaneBenchmarkCase("b", 0.4, 1.0 / 240.0, (0.02, 0.3)),
        SpherePlaneBenchmarkCase("c", 0.4, 1.0 / 240.0, (0.02, 0.3)),
    )
    dataset = ContactBenchmarkDataset(
        config={},
        mujoco_version="test",
        cases=tuple({"case_id": case.case_id} for case in cases),
        results=(
            _result(BenchmarkMode.FIXED_COARSE, validity=BenchmarkValidity.NONPHYSICAL_REBOUND),
            _result(BenchmarkMode.ADAPTIVE),
        ),
        comparisons=(
            _comparison("b", adaptive_e=0.3, adaptive_p=0.001),
            _comparison("a", adaptive_e=0.3, adaptive_p=0.003),
            _comparison("c", adaptive_e=0.1, adaptive_p=0.004),
        ),
        units={},
    )

    selected = select_convergence_cases(cases, dataset)

    assert [case.case_id for case in selected] == ["a", "b", "c"]


def test_failure_reason_priority_and_secondary_reasons() -> None:
    attribution = attribute_adaptive_failure(
        comparison=_comparison(),
        adaptive_result=_result(BenchmarkMode.ADAPTIVE, validity=BenchmarkValidity.NONPHYSICAL_REBOUND),
        trace=_trace(lead=0.2, limited=True, episodes=2),
        convergence=build_reference_convergence_result(
            "case",
            (_level(1, 0.1, 0.01), _level(2, 0.2, 0.02), _level(4, 0.4, 0.05)),
        ),
    )

    assert attribution.primary_reason is AdaptiveFailureReason.NONPHYSICAL_ADAPTIVE_RESULT
    assert AdaptiveFailureReason.REFERENCE_NOT_CONVERGED in attribution.secondary_reasons
    assert AdaptiveFailureReason.SHORT_PREDICTION_LEAD in attribution.secondary_reasons
    assert AdaptiveFailureReason.MAX_SUBSTEPS_LIMITED in attribution.secondary_reasons
    assert AdaptiveFailureReason.MULTIPLE_CONTACT_EPISODES in attribution.secondary_reasons


def test_both_acceptable_and_reference_unresolved_outcomes() -> None:
    converged = build_reference_convergence_result(
        "case",
        (_level(1, 0.5, 0.010), _level(2, 0.501, 0.0101), _level(4, 0.501, 0.0101)),
    )
    unresolved = build_reference_convergence_result(
        "case",
        (_level(1, 0.1, 0.01), _level(2, 0.2, 0.02), _level(4, 0.4, 0.05)),
    )

    acceptable = attribute_adaptive_failure(
        comparison=_comparison(coarse_e=0.001, adaptive_e=0.002, coarse_p=0.0001, adaptive_p=0.0002),
        adaptive_result=_result(BenchmarkMode.ADAPTIVE),
        trace=_trace(),
        convergence=converged,
    )
    unresolved_attr = attribute_adaptive_failure(
        comparison=_comparison(coarse_e=0.001, adaptive_e=0.002, coarse_p=0.0001, adaptive_p=0.0002),
        adaptive_result=_result(BenchmarkMode.ADAPTIVE),
        trace=_trace(),
        convergence=unresolved,
    )

    assert acceptable.restitution_outcome is ImprovementOutcome.BOTH_ACCEPTABLE
    assert acceptable.primary_reason is AdaptiveFailureReason.NONE
    assert unresolved_attr.restitution_outcome is ImprovementOutcome.REFERENCE_UNRESOLVED
    assert unresolved_attr.primary_reason is AdaptiveFailureReason.REFERENCE_NOT_CONVERGED


def test_late_prediction_when_contact_has_no_prediction() -> None:
    attribution = attribute_adaptive_failure(
        comparison=_comparison(),
        adaptive_result=_result(BenchmarkMode.ADAPTIVE),
        trace=_trace(lead=None),
        convergence=None,
    )

    assert attribution.primary_reason is AdaptiveFailureReason.LATE_PREDICTION


def test_early_fine_exit_reason_from_trace() -> None:
    base = _trace()
    episode = base.episodes[0]
    early_episode = AdaptiveContactEpisodeTrace(
        **{
            **episode.__dict__,
            "maximum_actual_substep_count": 1,
            "minimum_actual_timestep": 1.0 / 240.0,
            "state_at_maximum_penetration": ContactMotionState.RESTING,
        }
    )
    trace = AdaptiveDiagnosticTrace(
        **{
            **base.__dict__,
            "episodes": (early_episode,),
        }
    )
    attribution = attribute_adaptive_failure(
        comparison=_comparison(),
        adaptive_result=_result(BenchmarkMode.ADAPTIVE),
        trace=trace,
        convergence=None,
    )

    assert AdaptiveFailureReason.EARLY_FINE_EXIT in {attribution.primary_reason, *attribution.secondary_reasons}


def test_convergence_attribution_exports(tmp_path: Path) -> None:
    convergence = build_reference_convergence_result(
        "case",
        (_level(1, 0.5, 0.010), _level(2, 0.501, 0.0101), _level(4, 0.501, 0.0101)),
    )
    attribution = AdaptiveFailureAttribution(
        case_id="case",
        restitution_improved=True,
        penetration_improved=True,
        primary_reason=AdaptiveFailureReason.NONE,
        secondary_reasons=(),
        evidence=("ok",),
    )
    benchmark = ContactBenchmarkDataset(
        config={},
        mujoco_version="test",
        cases=({"case_id": "case"},),
        results=(),
        comparisons=(_comparison(),),
        units={},
    )
    trace = _trace()
    dataset = AttributionDiagnosticDataset(
        benchmark=benchmark,
        convergence=(convergence,),
        traces=(trace,),
        attributions=(attribution,),
        summary=AttributionBenchmarkSummary(
            total_cases=1,
            convergence_checked_cases=1,
            converged_reference_cases=1,
            unresolved_reference_cases=0,
            failure_reason_counts={"none": 1},
            improvement_outcome_counts={"not_applicable": 2},
            mean_prediction_lead_macro_steps=1.0,
            minimum_prediction_lead_macro_steps=1.0,
            max_substep_limited_case_count=0,
            multiple_episode_case_count=0,
            early_fine_exit_case_count=0,
        ),
        config={},
        units={},
    )

    export_convergence_csv((convergence,), tmp_path / "reference.csv")
    export_attribution_csv((attribution,), tmp_path / "attribution.csv")
    export_diagnostic_json(dataset, tmp_path / "diagnostics.json")
    write_attribution_markdown_report(dataset, tmp_path / "report.md")

    assert "overall_status" in (tmp_path / "reference.csv").read_text(encoding="utf8")
    assert "primary_reason" in (tmp_path / "attribution.csv").read_text(encoding="utf8")
    assert "reference_convergence" in (tmp_path / "diagnostics.json").read_text(encoding="utf8")
    assert "Failure Attribution" in (tmp_path / "report.md").read_text(encoding="utf8")
    assert "Overview" in build_attribution_markdown_report(dataset)
