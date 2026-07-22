"""Diagnostics for unresolved primary-impact reference convergence."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from physical_simulation.evaluation.adaptive_batch import (
    AdaptiveBatchCase,
    AdaptiveBatchConfig,
    AdaptiveBatchSceneType,
    ReferenceEvaluationMode,
    _to_benchmark_case,
)
from physical_simulation.evaluation.contact_benchmark import BenchmarkMode
from physical_simulation.evaluation.contact_convergence import (
    ReferenceConvergenceConfig,
    ReferenceConvergenceStatus,
    ReferenceMetricConvergence,
    _metric_convergence,
)
from physical_simulation.evaluation.contact_episode import (
    ContactEpisodeMetrics,
    ContactEpisodeSegmentationConfig,
    EpisodeMatchStatus,
    EpisodeMatchingConfig,
    collect_contact_episode_samples,
    match_contact_episodes,
    segment_contact_episodes,
)
from physical_simulation.mujoco import SubstepRecommendationConfig


class ReferenceUnresolvedReason(Enum):
    """Primary reason a checked reference case is not formally converged."""

    NONE = "none"
    EPISODE_UNMATCHED = "episode_unmatched"
    RESTITUTION_NOT_CONVERGED = "restitution_not_converged"
    PENETRATION_NOT_CONVERGED = "penetration_not_converged"
    DURATION_NOT_CONVERGED = "duration_not_converged"
    START_TIME_NOT_CONVERGED = "start_time_not_converged"
    NON_MONOTONIC_REFINEMENT = "non_monotonic_refinement"
    METRIC_SAMPLING_SENSITIVITY = "metric_sampling_sensitivity"
    INVALID_LEVEL = "invalid_level"
    UNKNOWN = "unknown"


class ReferenceDiagnosticStatus(Enum):
    """Report-level reference status."""

    NOT_CHECKED = "not_checked"
    CONVERGED = "converged"
    NEAR_CONVERGED = "near_converged"
    NOT_CONVERGED = "not_converged"
    INVALID = "invalid"


@dataclass(frozen=True)
class ReferenceMetricLevelDiagnostic:
    """One metric row at one refinement level."""

    case_id: str
    scene_type: AdaptiveBatchSceneType
    refinement_label: str
    refinement_factor: int
    timestep: float
    match_status: EpisodeMatchStatus
    episode_index: int | None
    restitution: float | None
    maximum_penetration: float | None
    contact_duration_seconds: float | None
    primary_start_time: float | None
    impact_speed: float | None
    separation_speed: float | None


@dataclass(frozen=True)
class ReferenceMetricDifferenceDiagnostic:
    """D1/D2/rho for one metric over three adjacent refinement levels."""

    metric_name: str
    d1: float | None
    d2: float | None
    rho: float | None
    absolute_tolerance: float
    relative_tolerance: float
    status: ReferenceConvergenceStatus
    near_converged: bool
    non_monotonic: bool


@dataclass(frozen=True)
class ReferenceCaseDiagnostic:
    """Diagnostics for one checked or not-checked case."""

    case: AdaptiveBatchCase
    status: ReferenceDiagnosticStatus
    base_status: ReferenceDiagnosticStatus
    extra_fine_status: ReferenceDiagnosticStatus | None
    primary_reason: ReferenceUnresolvedReason
    reasons: tuple[ReferenceUnresolvedReason, ...]
    blocking_metrics: tuple[str, ...]
    levels: tuple[ReferenceMetricLevelDiagnostic, ...]
    metric_differences: Mapping[str, ReferenceMetricDifferenceDiagnostic]
    extra_metric_differences: Mapping[str, ReferenceMetricDifferenceDiagnostic]
    extra_fine_added: bool


@dataclass(frozen=True)
class ReferenceDiagnosticsSummary:
    """Aggregate reference diagnostics."""

    total_case_count: int
    checked_case_count: int
    not_checked_case_count: int
    converged_case_count: int
    near_converged_case_count: int
    unresolved_case_count: int
    invalid_case_count: int
    extra_fine_case_count: int
    extra_fine_converged_case_count: int
    reason_counts: Mapping[str, int]
    blocking_metric_counts: Mapping[str, int]
    sphere_plane_unresolved_count: int
    sphere_sphere_unresolved_count: int
    high_speed_unresolved_count: int
    dt240_unresolved_count: int


@dataclass(frozen=True)
class ReferenceDiagnosticsDataset:
    """Exportable reference diagnostics dataset."""

    diagnostics: tuple[ReferenceCaseDiagnostic, ...]
    summary: ReferenceDiagnosticsSummary
    config: Mapping[str, object]
    mujoco_version: str | None
    git_commit: str | None


def run_reference_convergence_diagnostics(
    cases: Sequence[AdaptiveBatchCase],
    *,
    checked_case_ids: Sequence[str],
    batch_config: AdaptiveBatchConfig,
    add_extra_fine_for_unresolved: bool = True,
) -> ReferenceDiagnosticsDataset:
    """Run reference convergence diagnostics for selected checked cases."""
    checked = set(checked_case_ids)
    rows: list[ReferenceCaseDiagnostic] = []
    convergence_config = ReferenceConvergenceConfig(refinement_factors=batch_config.reference.refinement_factors)
    for case in cases:
        if case.case_id not in checked:
            rows.append(_not_checked(case))
            continue
        diagnostic = _diagnose_checked_case(case, batch_config=batch_config, convergence_config=convergence_config)
        if add_extra_fine_for_unresolved and diagnostic.status not in {
            ReferenceDiagnosticStatus.CONVERGED,
            ReferenceDiagnosticStatus.INVALID,
        }:
            diagnostic = _diagnose_checked_case(
                case,
                batch_config=batch_config,
                convergence_config=ReferenceConvergenceConfig(refinement_factors=(*convergence_config.refinement_factors, 8)),
                base_factor_count=len(convergence_config.refinement_factors),
            )
        rows.append(diagnostic)
    diagnostics = tuple(rows)
    summary = build_reference_diagnostics_summary(diagnostics)
    return ReferenceDiagnosticsDataset(
        diagnostics=diagnostics,
        summary=summary,
        config={
            "checked_case_ids": tuple(checked_case_ids),
            "refinement_factors": convergence_config.refinement_factors,
            "add_extra_fine_for_unresolved": add_extra_fine_for_unresolved,
            "reference_mode": batch_config.reference.mode.value if isinstance(batch_config.reference.mode, ReferenceEvaluationMode) else str(batch_config.reference.mode),
        },
        mujoco_version=_mujoco_version(),
        git_commit=_git_commit(),
    )


def build_reference_diagnostics_summary(
    diagnostics: Sequence[ReferenceCaseDiagnostic],
) -> ReferenceDiagnosticsSummary:
    """Summarize reference diagnostics."""
    checked = [item for item in diagnostics if item.status is not ReferenceDiagnosticStatus.NOT_CHECKED]
    unresolved = [item for item in checked if item.status is ReferenceDiagnosticStatus.NOT_CONVERGED]
    return ReferenceDiagnosticsSummary(
        total_case_count=len(diagnostics),
        checked_case_count=len(checked),
        not_checked_case_count=sum(item.status is ReferenceDiagnosticStatus.NOT_CHECKED for item in diagnostics),
        converged_case_count=sum(item.status is ReferenceDiagnosticStatus.CONVERGED for item in checked),
        near_converged_case_count=sum(item.status is ReferenceDiagnosticStatus.NEAR_CONVERGED for item in checked),
        unresolved_case_count=len(unresolved),
        invalid_case_count=sum(item.status is ReferenceDiagnosticStatus.INVALID for item in checked),
        extra_fine_case_count=sum(item.extra_fine_added for item in checked),
        extra_fine_converged_case_count=sum(item.extra_fine_status is ReferenceDiagnosticStatus.CONVERGED for item in checked),
        reason_counts=dict(sorted(Counter(item.primary_reason.value for item in checked).items())),
        blocking_metric_counts=dict(sorted(Counter(metric for item in checked for metric in item.blocking_metrics).items())),
        sphere_plane_unresolved_count=sum(item.case.scene_type is AdaptiveBatchSceneType.SPHERE_PLANE for item in unresolved),
        sphere_sphere_unresolved_count=sum(item.case.scene_type is AdaptiveBatchSceneType.SPHERE_SPHERE for item in unresolved),
        high_speed_unresolved_count=sum(_impact_speed_range(item) == "high" for item in unresolved),
        dt240_unresolved_count=sum(math.isclose(item.case.macro_timestep, 1.0 / 240.0, rel_tol=0.0, abs_tol=1.0e-12) for item in unresolved),
    )


def export_reference_diagnostics(
    dataset: ReferenceDiagnosticsDataset,
    output_dir: str | Path,
) -> Mapping[str, str]:
    """Export CSV, JSON, and Markdown reference diagnostics."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "metric_levels_csv": target / "metric_levels.csv",
        "unresolved_cases_csv": target / "unresolved_cases.csv",
        "diagnostics_json": target / "diagnostics.json",
        "report_markdown": target / "report.md",
    }
    _write_csv(paths["metric_levels_csv"], [_level_row(level) for item in dataset.diagnostics for level in item.levels])
    _write_csv(paths["unresolved_cases_csv"], [_case_row(item) for item in dataset.diagnostics if item.status in {ReferenceDiagnosticStatus.NOT_CONVERGED, ReferenceDiagnosticStatus.NEAR_CONVERGED, ReferenceDiagnosticStatus.INVALID}])
    paths["diagnostics_json"].write_text(json.dumps(_dataset_to_json(dataset), indent=2, sort_keys=True), encoding="utf8")
    paths["report_markdown"].write_text(build_reference_diagnostics_markdown_report(dataset), encoding="utf8")
    return {key: str(path) for key, path in paths.items()}


def build_reference_diagnostics_markdown_report(dataset: ReferenceDiagnosticsDataset) -> str:
    """Build a concise Markdown report."""
    summary = dataset.summary
    lines = [
        "# Reference Convergence Diagnostics",
        "",
        "## Executive Summary",
        "",
        f"- checked / total: {summary.checked_case_count} / {summary.total_case_count}",
        f"- converged / near-converged / unresolved / invalid: {summary.converged_case_count} / {summary.near_converged_case_count} / {summary.unresolved_case_count} / {summary.invalid_case_count}",
        f"- not checked: {summary.not_checked_case_count}",
        f"- extra-fine cases / newly converged: {summary.extra_fine_case_count} / {summary.extra_fine_converged_case_count}",
        f"- reason counts: {dict(summary.reason_counts)}",
        f"- blocking metric counts: {dict(summary.blocking_metric_counts)}",
        "",
        "## Unresolved Cases",
        "",
    ]
    for item in dataset.diagnostics:
        if item.status not in {ReferenceDiagnosticStatus.NOT_CONVERGED, ReferenceDiagnosticStatus.NEAR_CONVERGED, ReferenceDiagnosticStatus.INVALID}:
            continue
        lines.append(
            f"- {item.case.case_id}: status={item.status.value}, reason={item.primary_reason.value}, "
            f"blocking={list(item.blocking_metrics)}, extra_fine={None if item.extra_fine_status is None else item.extra_fine_status.value}"
        )
    lines.extend([
        "",
        "## Scene Differences",
        "",
        f"- sphere-plane unresolved: {summary.sphere_plane_unresolved_count}",
        f"- sphere-sphere unresolved: {summary.sphere_sphere_unresolved_count}",
        f"- high-impact-speed unresolved: {summary.high_speed_unresolved_count}",
        f"- macro timestep 1/240 unresolved: {summary.dt240_unresolved_count}",
        "",
        "## Notes",
        "",
        "- NOT_CHECKED cases did not run refinement and are not counted as unresolved.",
        "- NEAR_CONVERGED is diagnostic only and is not counted as formal convergence.",
        "- This report does not modify adaptive policy, prediction horizon, solref/solimp, segmentation, matching, or tolerances.",
    ])
    return "\n".join(lines) + "\n"


def _diagnose_checked_case(
    case: AdaptiveBatchCase,
    *,
    batch_config: AdaptiveBatchConfig,
    convergence_config: ReferenceConvergenceConfig,
    base_factor_count: int | None = None,
) -> ReferenceCaseDiagnostic:
    benchmark_case = _to_benchmark_case(case)
    levels: list[ReferenceMetricLevelDiagnostic] = []
    reference_primary: ContactEpisodeMetrics | None = None
    level_episodes: list[tuple[int, float, tuple[ContactEpisodeMetrics, ...]]] = []
    for index, factor in enumerate(convergence_config.refinement_factors):
        recommendation = SubstepRecommendationConfig(maximum_substeps=batch_config.recommendation.maximum_substeps * factor)
        timestep = case.macro_timestep / recommendation.maximum_substeps
        episodes = segment_contact_episodes(
            collect_contact_episode_samples(benchmark_case, mode=BenchmarkMode.FIXED_FINE, recommendation=recommendation),
            config=batch_config.episode_segmentation,
        )
        level_episodes.append((factor, timestep, episodes))
    if level_episodes:
        reference_primary = _primary(level_episodes[-1][2])
    for index, (factor, timestep, episodes) in enumerate(level_episodes):
        episode = _primary(episodes)
        match_status = EpisodeMatchStatus.UNMATCHED_REFERENCE
        if reference_primary is not None and episode is not None:
            match = match_contact_episodes(reference=(reference_primary,), comparison=(episode,), config=batch_config.episode_matching)
            match_status = match[0].status if match else EpisodeMatchStatus.UNMATCHED_REFERENCE
        levels.append(_level(case, index, factor, timestep, episode, match_status))

    base_count = base_factor_count or len(convergence_config.refinement_factors)
    base_levels = tuple(levels[:base_count])
    all_levels = tuple(levels)
    base_metrics = _metric_diagnostics(base_levels, convergence_config)
    status, reasons, blocking = _status_reasons(base_levels, base_metrics)
    extra_metrics = _metric_diagnostics(all_levels[-3:], convergence_config) if len(all_levels) > base_count else {}
    extra_status = None
    if len(all_levels) > base_count:
        extra_status, _, _ = _status_reasons(all_levels[-3:], extra_metrics)
    return ReferenceCaseDiagnostic(
        case=case,
        status=status if extra_status is None else status,
        base_status=status,
        extra_fine_status=extra_status,
        primary_reason=reasons[0] if reasons else ReferenceUnresolvedReason.NONE,
        reasons=tuple(reasons),
        blocking_metrics=tuple(blocking),
        levels=all_levels,
        metric_differences=base_metrics,
        extra_metric_differences=extra_metrics,
        extra_fine_added=len(all_levels) > base_count,
    )


def _metric_diagnostics(
    levels: Sequence[ReferenceMetricLevelDiagnostic],
    config: ReferenceConvergenceConfig,
) -> dict[str, ReferenceMetricDifferenceDiagnostic]:
    invalid = len(levels) < 3 or any(level.match_status is not EpisodeMatchStatus.MATCHED for level in levels[:-1])
    specs = {
        "restitution": ([level.restitution for level in levels], config.restitution_absolute_tolerance, config.restitution_relative_tolerance, True),
        "maximum_penetration": ([level.maximum_penetration for level in levels], config.penetration_absolute_tolerance, config.penetration_relative_tolerance, True),
        "contact_duration": ([level.contact_duration_seconds for level in levels], config.duration_absolute_tolerance, config.duration_relative_tolerance, False),
        "primary_start_time": ([level.primary_start_time for level in levels], config.duration_absolute_tolerance, config.duration_relative_tolerance, True),
        "impact_speed": ([level.impact_speed for level in levels], config.rebound_speed_absolute_tolerance, config.rebound_speed_relative_tolerance, True),
        "separation_speed": ([level.separation_speed for level in levels], config.rebound_speed_absolute_tolerance, config.rebound_speed_relative_tolerance, False),
    }
    rows = {}
    for name, (values, abs_tol, rel_tol, required) in specs.items():
        metric = _metric_convergence(name, values, abs_tol, rel_tol, invalid=invalid, required=required)
        near = _near_converged(metric, values)
        rows[name] = ReferenceMetricDifferenceDiagnostic(
            metric_name=name,
            d1=metric.coarse_to_fine_difference,
            d2=metric.fine_to_finer_difference,
            rho=metric.difference_ratio,
            absolute_tolerance=metric.absolute_tolerance,
            relative_tolerance=metric.relative_tolerance,
            status=metric.status,
            near_converged=near,
            non_monotonic=False if metric.difference_ratio is None else metric.difference_ratio > 1.0,
        )
    return rows


def _status_reasons(
    levels: Sequence[ReferenceMetricLevelDiagnostic],
    metrics: Mapping[str, ReferenceMetricDifferenceDiagnostic],
) -> tuple[ReferenceDiagnosticStatus, list[ReferenceUnresolvedReason], list[str]]:
    if len(levels) < 3 or any(_level_invalid(level) for level in levels):
        return ReferenceDiagnosticStatus.INVALID, [ReferenceUnresolvedReason.INVALID_LEVEL], ()
    if any(level.match_status is not EpisodeMatchStatus.MATCHED for level in levels[:-1]):
        return ReferenceDiagnosticStatus.INVALID, [ReferenceUnresolvedReason.EPISODE_UNMATCHED], ()
    required = ("restitution", "maximum_penetration", "primary_start_time", "impact_speed")
    optional = ("contact_duration", "separation_speed")
    blocking = [
        name
        for name in (*required, *optional)
        if metrics[name].status is not ReferenceConvergenceStatus.CONVERGED
    ]
    if not blocking:
        return ReferenceDiagnosticStatus.CONVERGED, [ReferenceUnresolvedReason.NONE], ()
    reasons: list[ReferenceUnresolvedReason] = []
    if any(metrics[name].non_monotonic for name in blocking):
        reasons.append(ReferenceUnresolvedReason.NON_MONOTONIC_REFINEMENT)
    reason_by_metric = {
        "restitution": ReferenceUnresolvedReason.RESTITUTION_NOT_CONVERGED,
        "maximum_penetration": ReferenceUnresolvedReason.PENETRATION_NOT_CONVERGED,
        "contact_duration": ReferenceUnresolvedReason.DURATION_NOT_CONVERGED,
        "primary_start_time": ReferenceUnresolvedReason.START_TIME_NOT_CONVERGED,
    }
    for name in blocking:
        if name in reason_by_metric:
            reasons.append(reason_by_metric[name])
    if "maximum_penetration" in blocking or "contact_duration" in blocking:
        reasons.append(ReferenceUnresolvedReason.METRIC_SAMPLING_SENSITIVITY)
    if not reasons:
        reasons.append(ReferenceUnresolvedReason.UNKNOWN)
    near = len(blocking) <= 1 and all(metrics[name].near_converged for name in blocking)
    return ReferenceDiagnosticStatus.NEAR_CONVERGED if near else ReferenceDiagnosticStatus.NOT_CONVERGED, _dedupe(reasons), blocking


def _near_converged(metric: ReferenceMetricConvergence, values: Sequence[float | None]) -> bool:
    if metric.status is ReferenceConvergenceStatus.CONVERGED:
        return True
    if metric.fine_to_finer_difference is None or len(values) < 3 or values[-1] is None or values[-2] is None:
        return False
    scale = max(abs(values[-1]), abs(values[-2]), 1.0e-12)
    threshold = max(metric.absolute_tolerance, metric.relative_tolerance * scale)
    return metric.fine_to_finer_difference <= 2.0 * threshold


def _not_checked(case: AdaptiveBatchCase) -> ReferenceCaseDiagnostic:
    return ReferenceCaseDiagnostic(
        case=case,
        status=ReferenceDiagnosticStatus.NOT_CHECKED,
        base_status=ReferenceDiagnosticStatus.NOT_CHECKED,
        extra_fine_status=None,
        primary_reason=ReferenceUnresolvedReason.NONE,
        reasons=(ReferenceUnresolvedReason.NONE,),
        blocking_metrics=(),
        levels=(),
        metric_differences={},
        extra_metric_differences={},
        extra_fine_added=False,
    )


def _level(
    case: AdaptiveBatchCase,
    index: int,
    factor: int,
    timestep: float,
    episode: ContactEpisodeMetrics | None,
    match_status: EpisodeMatchStatus,
) -> ReferenceMetricLevelDiagnostic:
    labels = {0: "fine", 1: "finer", 2: "ultra_fine", 3: "extra_fine"}
    return ReferenceMetricLevelDiagnostic(
        case_id=case.case_id,
        scene_type=case.scene_type,
        refinement_label=labels.get(index, f"level_{index}"),
        refinement_factor=factor,
        timestep=timestep,
        match_status=match_status,
        episode_index=None if episode is None else episode.episode_index,
        restitution=None if episode is None else episode.restitution,
        maximum_penetration=None if episode is None else episode.maximum_penetration,
        contact_duration_seconds=None if episode is None else episode.duration_seconds,
        primary_start_time=None if episode is None else episode.start_time,
        impact_speed=None if episode is None else episode.impact_speed,
        separation_speed=None if episode is None else episode.separation_speed,
    )


def _primary(episodes: Sequence[ContactEpisodeMetrics]) -> ContactEpisodeMetrics | None:
    for episode in episodes:
        if episode.kind.value == "primary_impact":
            return episode
    return episodes[0] if episodes else None


def _level_invalid(level: ReferenceMetricLevelDiagnostic) -> bool:
    return level.episode_index is None or level.maximum_penetration is None or not math.isfinite(level.maximum_penetration)


def _dedupe(reasons: Sequence[ReferenceUnresolvedReason]) -> list[ReferenceUnresolvedReason]:
    rows = []
    for reason in reasons:
        if reason not in rows:
            rows.append(reason)
    return rows


def _impact_speed_range(item: ReferenceCaseDiagnostic) -> str:
    speeds = [level.impact_speed for level in item.levels if level.impact_speed is not None]
    if not speeds:
        return "unknown"
    speed = speeds[-1]
    if speed < 2.0:
        return "low"
    if speed < 4.0:
        return "medium"
    return "high"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _level_row(level: ReferenceMetricLevelDiagnostic) -> dict[str, object]:
    return _serializable(level)


def _case_row(item: ReferenceCaseDiagnostic) -> dict[str, object]:
    row = {
        "case_id": item.case.case_id,
        "scene_type": item.case.scene_type.value,
        "macro_timestep": item.case.macro_timestep,
        "status": item.status.value,
        "base_status": item.base_status.value,
        "extra_fine_status": None if item.extra_fine_status is None else item.extra_fine_status.value,
        "primary_reason": item.primary_reason.value,
        "reasons": ";".join(reason.value for reason in item.reasons),
        "blocking_metrics": ";".join(item.blocking_metrics),
        "extra_fine_added": item.extra_fine_added,
    }
    for prefix, metrics in (("base", item.metric_differences), ("extra", item.extra_metric_differences)):
        for name, metric in metrics.items():
            row[f"{prefix}_{name}_d1"] = metric.d1
            row[f"{prefix}_{name}_d2"] = metric.d2
            row[f"{prefix}_{name}_rho"] = metric.rho
            row[f"{prefix}_{name}_status"] = metric.status.value
            row[f"{prefix}_{name}_near"] = metric.near_converged
    return row


def _dataset_to_json(dataset: ReferenceDiagnosticsDataset) -> dict[str, object]:
    return {
        "schema_version": "reference-diagnostics/v1",
        "mujoco_version": dataset.mujoco_version,
        "git_commit": dataset.git_commit,
        "config": _serializable(dataset.config),
        "summary": _serializable(dataset.summary),
        "diagnostics": [_serializable(item) for item in dataset.diagnostics],
        "units": {
            "time": "seconds",
            "length": "meters",
            "velocity": "meters/second",
            "rho": "D2 / D1",
        },
    }


def _serializable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _serializable(val) for key, val in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _serializable(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    return str(value)


def _mujoco_version() -> str | None:
    try:
        import mujoco

        return getattr(mujoco, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    except Exception:  # noqa: BLE001
        return None
    return result.stdout.strip() or None
