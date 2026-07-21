"""Integration tests for the adaptive primary-impact batch pipeline."""

from __future__ import annotations

from pathlib import Path

from physical_simulation.evaluation import (
    BatchReferenceConfig,
    ReferenceEvaluationStatus,
    ReferenceEvaluationMode,
    make_smoke_adaptive_batch,
    run_adaptive_primary_batch,
)


def test_smoke_adaptive_primary_batch_pipeline_exports_artifacts(tmp_path: Path) -> None:
    cases, config = make_smoke_adaptive_batch()
    cases = cases[:3]
    config = _with_output(config, tmp_path)
    config = _with_reference_limit(config, 2)
    result = run_adaptive_primary_batch(cases, config=config)

    assert result.summary.total_case_count == 3
    assert result.summary.invalid_case_count == 0
    assert result.summary.primary_matched_case_count >= 1
    assert result.summary.reference_checked_case_count >= 1
    assert result.summary.reference_not_checked_case_count >= 1
    assert any(item.reference_evaluation_status is ReferenceEvaluationStatus.NOT_CHECKED for item in result.results)
    assert any(item.reference_evaluation_status is not ReferenceEvaluationStatus.NOT_CHECKED for item in result.results)
    assert result.summary.primary_restitution_improvement_denominator <= result.summary.reference_converged_case_count
    assert set(result.artifact_paths) == {
        "accuracy_cost_csv",
        "cases_csv",
        "diagnostics_json",
        "group_summary_csv",
        "primary_results_csv",
        "reference_convergence_csv",
        "report_markdown",
    }
    for path in result.artifact_paths.values():
        assert Path(path).exists()
    assert '"schema_version": "adaptive-primary-batch/v1"' in (tmp_path / "diagnostics.json").read_text(encoding="utf8")


def test_smoke_batch_is_deterministic_with_same_seed(tmp_path: Path) -> None:
    cases_a, config_a = make_smoke_adaptive_batch()
    cases_b, config_b = make_smoke_adaptive_batch()
    assert [case.case_id for case in cases_a] == [case.case_id for case in cases_b]

    result_a = run_adaptive_primary_batch(cases_a[:3], config=_with_reference_limit(_with_output(config_a, tmp_path / "a"), 2))
    result_b = run_adaptive_primary_batch(cases_b[:3], config=_with_reference_limit(_with_output(config_b, tmp_path / "b"), 2))
    assert result_a.selected_reference_case_ids == result_b.selected_reference_case_ids
    assert result_a.summary.primary_matched_case_count == result_b.summary.primary_matched_case_count
    assert result_a.summary.reference_checked_case_count == result_b.summary.reference_checked_case_count


def _with_output(config, output_dir: Path):
    from dataclasses import replace

    return replace(config, output_dir=output_dir)


def _with_reference_limit(config, maximum_selected_cases: int):
    from dataclasses import replace

    return replace(
        config,
        reference=BatchReferenceConfig(
            mode=ReferenceEvaluationMode.SELECTED,
            maximum_selected_cases=maximum_selected_cases,
            top_k_adaptive_restitution_error=1,
            top_k_adaptive_penetration_error=1,
            top_k_adaptive_duration_error=1,
        ),
    )
