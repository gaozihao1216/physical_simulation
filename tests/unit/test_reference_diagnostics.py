"""Unit tests for reference convergence diagnostics."""

from __future__ import annotations

from pathlib import Path

from physical_simulation.evaluation.adaptive_batch import (
    AdaptiveBatchSceneType,
    BatchGenerationConfig,
    generate_sphere_plane_batch_cases,
)
from physical_simulation.evaluation.contact_convergence import ReferenceConvergenceConfig
from physical_simulation.evaluation.contact_episode import EpisodeMatchStatus
from physical_simulation.evaluation.reference_diagnostics import (
    ReferenceCaseDiagnostic,
    ReferenceDiagnosticStatus,
    ReferenceDiagnosticsDataset,
    ReferenceMetricLevelDiagnostic,
    ReferenceUnresolvedReason,
    _metric_diagnostics,
    _status_reasons,
    build_reference_diagnostics_summary,
    export_reference_diagnostics,
)


def test_metric_diagnostics_compute_d1_d2_rho_and_converged() -> None:
    levels = (
        _level("fine", 1, restitution=0.40, penetration=0.010, start=0.10),
        _level("finer", 2, restitution=0.405, penetration=0.0096, start=0.101),
        _level("ultra_fine", 4, restitution=0.406, penetration=0.0095, start=0.1012),
    )
    metrics = _metric_diagnostics(levels, ReferenceConvergenceConfig())
    assert metrics["restitution"].d1 == 0.0050000000000000044
    assert metrics["restitution"].d2 == 0.0010000000000000009
    assert metrics["restitution"].rho is not None


def test_status_reasons_detect_episode_unmatched() -> None:
    levels = (
        _level("fine", 1, match=EpisodeMatchStatus.MATCHED),
        _level("finer", 2, match=EpisodeMatchStatus.UNMATCHED_REFERENCE),
        _level("ultra_fine", 4, match=EpisodeMatchStatus.MATCHED),
    )
    status, reasons, blocking = _status_reasons(levels, _metric_diagnostics(levels, ReferenceConvergenceConfig()))
    assert status is ReferenceDiagnosticStatus.INVALID
    assert reasons == [ReferenceUnresolvedReason.EPISODE_UNMATCHED]
    assert blocking == ()


def test_status_reasons_detect_non_monotonic_penetration() -> None:
    levels = (
        _level("fine", 1, restitution=0.4, penetration=0.0100, start=0.10),
        _level("finer", 2, restitution=0.401, penetration=0.0098, start=0.1001),
        _level("ultra_fine", 4, restitution=0.402, penetration=0.0105, start=0.1002),
    )
    status, reasons, blocking = _status_reasons(levels, _metric_diagnostics(levels, ReferenceConvergenceConfig()))
    assert status is ReferenceDiagnosticStatus.NOT_CONVERGED
    assert ReferenceUnresolvedReason.NON_MONOTONIC_REFINEMENT in reasons
    assert "maximum_penetration" in blocking


def test_summary_distinguishes_near_converged_and_unresolved() -> None:
    case = generate_sphere_plane_batch_cases(config=BatchGenerationConfig(maximum_case_count=1))[0]
    diagnostics = (
        _case(case, ReferenceDiagnosticStatus.CONVERGED, ReferenceUnresolvedReason.NONE, ()),
        _case(case, ReferenceDiagnosticStatus.NEAR_CONVERGED, ReferenceUnresolvedReason.PENETRATION_NOT_CONVERGED, ("maximum_penetration",)),
        _case(case, ReferenceDiagnosticStatus.NOT_CONVERGED, ReferenceUnresolvedReason.NON_MONOTONIC_REFINEMENT, ("restitution",)),
        _case(case, ReferenceDiagnosticStatus.NOT_CHECKED, ReferenceUnresolvedReason.NONE, ()),
    )
    summary = build_reference_diagnostics_summary(diagnostics)
    assert summary.checked_case_count == 3
    assert summary.not_checked_case_count == 1
    assert summary.converged_case_count == 1
    assert summary.near_converged_case_count == 1
    assert summary.unresolved_case_count == 1


def test_export_reference_diagnostics_writes_all_files(tmp_path: Path) -> None:
    case = generate_sphere_plane_batch_cases(config=BatchGenerationConfig(maximum_case_count=1))[0]
    diagnostic = _case(case, ReferenceDiagnosticStatus.NOT_CONVERGED, ReferenceUnresolvedReason.RESTITUTION_NOT_CONVERGED, ("restitution",))
    dataset = ReferenceDiagnosticsDataset(
        diagnostics=(diagnostic,),
        summary=build_reference_diagnostics_summary((diagnostic,)),
        config={"checked_case_ids": (case.case_id,)},
        mujoco_version="test",
        git_commit=None,
    )
    paths = export_reference_diagnostics(dataset, tmp_path)
    assert set(paths) == {"metric_levels_csv", "unresolved_cases_csv", "diagnostics_json", "report_markdown"}
    assert '"schema_version": "reference-diagnostics/v1"' in (tmp_path / "diagnostics.json").read_text(encoding="utf8")
    assert "Reference Convergence Diagnostics" in (tmp_path / "report.md").read_text(encoding="utf8")


def _level(
    label: str,
    factor: int,
    *,
    restitution: float = 0.4,
    penetration: float = 0.01,
    start: float = 0.1,
    match: EpisodeMatchStatus = EpisodeMatchStatus.MATCHED,
) -> ReferenceMetricLevelDiagnostic:
    return ReferenceMetricLevelDiagnostic(
        case_id="case",
        scene_type=AdaptiveBatchSceneType.SPHERE_PLANE,
        refinement_label=label,
        refinement_factor=factor,
        timestep=1.0 / (240.0 * factor),
        match_status=match,
        episode_index=0,
        restitution=restitution,
        maximum_penetration=penetration,
        contact_duration_seconds=0.01,
        primary_start_time=start,
        impact_speed=3.0,
        separation_speed=1.0,
    )


def _case(case, status, reason, blocking) -> ReferenceCaseDiagnostic:
    levels = (_level("fine", 1), _level("finer", 2), _level("ultra_fine", 4))
    return ReferenceCaseDiagnostic(
        case=case,
        status=status,
        base_status=status,
        extra_fine_status=None,
        primary_reason=reason,
        reasons=(reason,),
        blocking_metrics=tuple(blocking),
        levels=levels,
        metric_differences=_metric_diagnostics(levels, ReferenceConvergenceConfig()),
        extra_metric_differences={},
        extra_fine_added=False,
    )

