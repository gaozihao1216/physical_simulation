"""Unit tests for adaptive primary-impact batch orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from physical_simulation.evaluation.adaptive_batch import (
    AccuracyCostPoint,
    AdaptiveBatchCase,
    AdaptiveBatchCaseResult,
    AdaptiveBatchConfig,
    AdaptiveBatchSceneType,
    BatchGenerationConfig,
    BatchReferenceConfig,
    ReferenceEvaluationMode,
    ReferenceEvaluationStatus,
    build_accuracy_cost_points,
    build_adaptive_batch_group_summaries,
    build_adaptive_batch_summary,
    export_adaptive_batch_csvs,
    find_nondominated_accuracy_cost_points,
    generate_sphere_plane_batch_cases,
    generate_sphere_sphere_batch_cases,
    make_smoke_adaptive_batch,
    run_adaptive_primary_batch,
)
from physical_simulation.evaluation.adaptive_batch import _select_reference_case_ids
from physical_simulation.evaluation.contact_benchmark import BenchmarkValidity
from physical_simulation.evaluation.contact_convergence import AdaptiveFailureReason, ImprovementOutcome, ReferenceConvergenceStatus
from physical_simulation.evaluation.contact_episode import EpisodeMatchStatus
from physical_simulation.evaluation.primary_attribution import (
    AttributionScope,
    PrimaryImpactCaseOutcome,
    PrimaryImpactFailureAttribution,
)
from physical_simulation.mujoco import AdaptiveSubstepConfig, AnalyticPlane, MuJoCoContactSolverParams
from physical_simulation.validation.errors import PhysicsValidationError


def test_batch_case_validation_requires_scene_specific_fields() -> None:
    with pytest.raises(PhysicsValidationError):
        AdaptiveBatchCase(
            case_id="bad",
            scene_type=AdaptiveBatchSceneType.SPHERE_PLANE,
            macro_timestep=1.0 / 240.0,
            total_simulation_time=0.5,
            contact_params=MuJoCoContactSolverParams(solref=(0.02, 0.3), solimp=(0.9, 0.9, 0.001, 0.5, 2.0)),
            adaptive_config=AdaptiveSubstepConfig(),
            sphere_a_radius=0.1,
            sphere_a_mass=1.0,
            sphere_a_initial_position=(0.0, 0.0, 1.0),
            sphere_a_initial_velocity=(0.0, 0.0, 0.0),
        )


def test_case_id_uniqueness_is_enforced() -> None:
    case = generate_sphere_plane_batch_cases(config=BatchGenerationConfig(maximum_case_count=1))[0]
    with pytest.raises(PhysicsValidationError):
        run_adaptive_primary_batch((case, case), config=AdaptiveBatchConfig(reference=BatchReferenceConfig(mode=ReferenceEvaluationMode.NONE)))


def test_deterministic_generation_is_stable_and_layered() -> None:
    config = BatchGenerationConfig(maximum_case_count=10, sampling_seed=3)
    first = generate_sphere_plane_batch_cases(config=config)
    second = generate_sphere_plane_batch_cases(config=config)
    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert len({case.metadata["height"] for case in first}) > 1
    assert len({case.macro_timestep for case in first}) > 1
    assert len({case.sphere_a_mass for case in first}) > 1


def test_sphere_sphere_generation_covers_required_motion_variants() -> None:
    cases = generate_sphere_sphere_batch_cases(config=BatchGenerationConfig(maximum_case_count=6))
    ids = [case.case_id for case in cases]
    assert any("symmetric_equal_mass" in case_id for case_id in ids)
    assert any("different_mass" in case_id for case_id in ids)
    assert any("one_static_initially" in case_id for case_id in ids)
    assert len({abs(case.sphere_a_initial_velocity[0] - case.sphere_b_initial_velocity[0]) for case in cases if case.sphere_b_initial_velocity}) > 1


def test_not_checked_summary_is_not_not_converged() -> None:
    case = generate_sphere_plane_batch_cases(config=BatchGenerationConfig(maximum_case_count=1))[0]
    result = AdaptiveBatchCaseResult(
        case=case,
        coarse_run=None,
        fine_run=None,
        adaptive_run=None,
        coarse_episodes=(),
        fine_episodes=(),
        adaptive_episodes=(),
        coarse_primary_match=None,
        adaptive_primary_match=None,
        provisional_primary_comparison=None,
        reference_evaluation_status=ReferenceEvaluationStatus.NOT_CHECKED,
        primary_reference_convergence=None,
        converged_reference_episode=None,
        primary_attribution=None,
        adaptive_trace=None,
        run_level_comparison=None,
        run_level_reference_convergence=None,
        error=None,
    )
    summary = build_adaptive_batch_summary((result,))
    assert summary.reference_not_checked_case_count == 1
    assert summary.reference_unresolved_case_count == 0


def test_improvement_denominator_filters_unchecked_and_unresolved() -> None:
    case = generate_sphere_plane_batch_cases(config=BatchGenerationConfig(maximum_case_count=1))[0]
    checked = _result_with_attr(
        case,
        ReferenceEvaluationStatus.CONVERGED,
        PrimaryImpactFailureAttribution(
            case_id=case.case_id,
            candidate_id="candidate",
            scope=AttributionScope.PRIMARY_IMPACT,
            case_outcome=PrimaryImpactCaseOutcome.IMPROVED,
            primary_reason=AdaptiveFailureReason.NONE,
            secondary_reasons=(),
            restitution_outcome=ImprovementOutcome.IMPROVED,
            penetration_outcome=ImprovementOutcome.IMPROVED,
            duration_outcome=ImprovementOutcome.IMPROVED,
            primary_reference_status=ReferenceConvergenceStatus.CONVERGED,
            run_level_reference_status=None,
            coarse_primary_match_status=None,
            adaptive_primary_match_status=EpisodeMatchStatus.MATCHED,
            evidence=(),
            adaptive_restitution_error=0.1,
            adaptive_penetration_error=0.01,
            adaptive_duration_error=0.001,
            adaptive_step_saving=0.9,
        ),
    )
    unchecked = _result_with_attr(case, ReferenceEvaluationStatus.NOT_CHECKED, None)
    summary = build_adaptive_batch_summary((checked, unchecked))
    assert summary.primary_restitution_improvement_numerator == 1
    assert summary.primary_restitution_improvement_denominator == 1


def test_group_summary_and_accuracy_cost_helpers() -> None:
    case = generate_sphere_plane_batch_cases(config=BatchGenerationConfig(maximum_case_count=1))[0]
    result = _result_with_attr(case, ReferenceEvaluationStatus.CONVERGED, None)
    groups = build_adaptive_batch_group_summaries((result,))
    assert {row.group_name for row in groups} >= {"scene_type", "macro_timestep", "solref", "impact_speed_range"}
    assert build_accuracy_cost_points((result,)) == ()


def test_selected_reference_strategy_includes_error_nonphysical_and_sphere_sphere() -> None:
    sphere_plane = generate_sphere_plane_batch_cases(config=BatchGenerationConfig(maximum_case_count=3))
    sphere_sphere = generate_sphere_sphere_batch_cases(config=BatchGenerationConfig(maximum_case_count=1))[0]
    results = (
        _selection_result(sphere_plane[0], validity=BenchmarkValidity.NONPHYSICAL_REBOUND, restitution_error=0.01),
        _selection_result(sphere_plane[1], validity=BenchmarkValidity.VALID, restitution_error=0.50),
        _selection_result(sphere_plane[2], validity=BenchmarkValidity.VALID, restitution_error=0.02),
        _selection_result(sphere_sphere, validity=BenchmarkValidity.VALID, restitution_error=0.01),
    )
    selected = _select_reference_case_ids(
        results,
        BatchReferenceConfig(maximum_selected_cases=None, top_k_adaptive_restitution_error=1, top_k_adaptive_penetration_error=0, top_k_adaptive_duration_error=0),
    )
    assert sphere_plane[0].case_id in selected
    assert sphere_plane[1].case_id in selected
    assert sphere_sphere.case_id in selected


def test_nondominated_filtering_is_stable() -> None:
    points = (
        _point("a", restitution_error=0.1, ratio=0.2),
        _point("b", restitution_error=0.2, ratio=0.3),
        _point("c", restitution_error=0.05, ratio=0.4),
    )
    kept = find_nondominated_accuracy_cost_points(points, error_function=lambda point: point.restitution_error or 0.0)
    assert [point.case_id for point in kept] == ["c", "a"]


def test_export_writes_csv_files(tmp_path: Path) -> None:
    case = generate_sphere_plane_batch_cases(config=BatchGenerationConfig(maximum_case_count=1))[0]
    result = _result_with_attr(case, ReferenceEvaluationStatus.NOT_CHECKED, None)
    paths = export_adaptive_batch_csvs((result,), build_adaptive_batch_group_summaries((result,)), (), tmp_path)
    assert set(paths) == {"cases_csv", "primary_results_csv", "group_summary_csv", "accuracy_cost_csv", "reference_convergence_csv"}
    assert (tmp_path / "cases.csv").read_text(encoding="utf8").startswith("adaptive_config")


def test_batch_failure_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    cases, config = make_smoke_adaptive_batch()
    cases = cases[:2]

    def fake_run(case, batch_config):
        if case.case_id == cases[0].case_id:
            raise RuntimeError("synthetic failure")
        return _result_with_attr(case, ReferenceEvaluationStatus.NOT_CHECKED, None)

    monkeypatch.setattr("physical_simulation.evaluation.adaptive_batch._run_provisional_case", fake_run)
    result = run_adaptive_primary_batch(cases, config=AdaptiveBatchConfig(reference=BatchReferenceConfig(mode=ReferenceEvaluationMode.NONE), output_dir=Path("artifacts/test_unused"), export_csv=False, export_json=False, export_markdown=False))
    assert result.summary.completed_case_count == 1
    assert result.summary.invalid_case_count == 1


def _result_with_attr(
    case: AdaptiveBatchCase,
    status: ReferenceEvaluationStatus,
    attr: PrimaryImpactFailureAttribution | None,
) -> AdaptiveBatchCaseResult:
    return AdaptiveBatchCaseResult(
        case=case,
        coarse_run=None,
        fine_run=None,
        adaptive_run=None,
        coarse_episodes=(),
        fine_episodes=(),
        adaptive_episodes=(),
        coarse_primary_match=None,
        adaptive_primary_match=None,
        provisional_primary_comparison=None,
        reference_evaluation_status=status,
        primary_reference_convergence=None,
        converged_reference_episode=None,
        primary_attribution=attr,
        adaptive_trace=None,
        run_level_comparison=None,
        run_level_reference_convergence=None,
        error=None,
    )


def _selection_result(
    case: AdaptiveBatchCase,
    *,
    validity: BenchmarkValidity,
    restitution_error: float,
) -> AdaptiveBatchCaseResult:
    return AdaptiveBatchCaseResult(
        case=case,
        coarse_run=SimpleNamespace(validity=validity),
        fine_run=None,
        adaptive_run=None,
        coarse_episodes=(),
        fine_episodes=(),
        adaptive_episodes=(),
        coarse_primary_match=None,
        adaptive_primary_match=None,
        provisional_primary_comparison=SimpleNamespace(
            adaptive_restitution_error=restitution_error,
            adaptive_penetration_error=0.0,
            adaptive_duration_error=0.0,
            adaptive_improves_restitution=True,
            adaptive_improves_penetration=True,
        ),
        reference_evaluation_status=ReferenceEvaluationStatus.NOT_CHECKED,
        primary_reference_convergence=None,
        converged_reference_episode=None,
        primary_attribution=None,
        adaptive_trace=None,
        run_level_comparison=None,
        run_level_reference_convergence=None,
        error=None,
    )


def _point(case_id: str, *, restitution_error: float, ratio: float) -> AccuracyCostPoint:
    return AccuracyCostPoint(
        case_id=case_id,
        scene_type=AdaptiveBatchSceneType.SPHERE_PLANE,
        restitution_error=restitution_error,
        penetration_error=None,
        duration_error=None,
        adaptive_step_ratio=ratio,
        adaptive_step_saving=1.0 - ratio,
        maximum_substep_count=4,
        substepped_macro_step_ratio=0.1,
    )
