"""Primary-impact failure attribution built on episode-level diagnostics."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from physical_simulation.evaluation.contact_benchmark import BenchmarkComparison
from physical_simulation.evaluation.contact_convergence import (
    AdaptiveDiagnosticTrace,
    AdaptiveFailureReason,
    ImprovementOutcome,
    ReferenceConvergenceResult,
    ReferenceConvergenceStatus,
)
from physical_simulation.evaluation.contact_episode import (
    ContactEpisodeKind,
    ContactEpisodeMetrics,
    EpisodeMatch,
    EpisodeMatchStatus,
    EpisodeReferenceConvergenceResult,
    PrimaryImpactBenchmarkComparison,
)
from physical_simulation.validation.asset_validator import _finite_float
from physical_simulation.validation.errors import PhysicsValidationError


class AttributionScope(Enum):
    """Scope used by failure attribution."""

    PRIMARY_IMPACT = "primary_impact"
    RUN_LEVEL = "run_level"
    FALLBACK_RUN_LEVEL = "fallback_run_level"
    UNAVAILABLE = "unavailable"


class PrimaryImpactCaseOutcome(Enum):
    """Case-level primary impact improvement outcome."""

    IMPROVED = "improved"
    PARTIALLY_IMPROVED = "partially_improved"
    NOT_IMPROVED = "not_improved"
    BOTH_ACCEPTABLE = "both_acceptable"
    REFERENCE_UNRESOLVED = "reference_unresolved"
    EPISODE_UNMATCHED = "episode_unmatched"
    INVALID_ADAPTIVE = "invalid_adaptive"


@dataclass(frozen=True)
class PrimaryImpactImprovementConfig:
    """Tolerances for primary impact metric outcomes."""

    restitution_error_tolerance: float = 0.005
    penetration_error_tolerance: float = 5.0e-4
    duration_error_tolerance: float = 1.0e-3

    def __post_init__(self) -> None:
        for field_name in (
            "restitution_error_tolerance",
            "penetration_error_tolerance",
            "duration_error_tolerance",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_float(
                    getattr(self, field_name),
                    field_name=field_name,
                    minimum=0.0,
                    error_type=PhysicsValidationError,
                ),
            )


@dataclass(frozen=True)
class AdaptiveAttributionConfig:
    """Configuration for assigning primary-impact failure reasons."""

    short_prediction_lead_macro_steps: float = 0.5
    samples_per_characteristic_time: int = 16


@dataclass(frozen=True)
class PrimaryImpactAttributionInput:
    """All structured inputs needed for primary-impact attribution."""

    case_id: str
    candidate_id: str
    coarse_episode: ContactEpisodeMetrics | None
    adaptive_episode: ContactEpisodeMetrics | None
    reference_episode: ContactEpisodeMetrics | None
    coarse_match: EpisodeMatch | None
    adaptive_match: EpisodeMatch | None
    primary_comparison: PrimaryImpactBenchmarkComparison | None
    primary_reference_convergence: EpisodeReferenceConvergenceResult | None
    run_level_comparison: BenchmarkComparison | None
    run_level_reference_convergence: ReferenceConvergenceResult | None
    adaptive_trace: AdaptiveDiagnosticTrace | None


@dataclass(frozen=True)
class PrimaryImpactFailureAttribution:
    """Failure attribution result scoped to primary impact when possible."""

    case_id: str
    candidate_id: str
    scope: AttributionScope
    case_outcome: PrimaryImpactCaseOutcome
    primary_reason: AdaptiveFailureReason
    secondary_reasons: tuple[AdaptiveFailureReason, ...]
    restitution_outcome: ImprovementOutcome
    penetration_outcome: ImprovementOutcome
    duration_outcome: ImprovementOutcome
    primary_reference_status: ReferenceConvergenceStatus | None
    run_level_reference_status: ReferenceConvergenceStatus | None
    coarse_primary_match_status: EpisodeMatchStatus | None
    adaptive_primary_match_status: EpisodeMatchStatus | None
    evidence: tuple[str, ...]
    adaptive_restitution_error: float | None = None
    adaptive_penetration_error: float | None = None
    adaptive_duration_error: float | None = None
    adaptive_step_saving: float | None = None
    prediction_lead_macro_steps: float | None = None
    minimum_timestep: float | None = None
    maximum_substep_count: int | None = None
    secondary_episode_count: int = 0
    chatter_count: int = 0


@dataclass(frozen=True)
class RunPrimaryAttributionDifference:
    """Difference between run-level and primary-impact attribution outcomes."""

    case_id: str
    run_level_restitution_outcome: ImprovementOutcome
    primary_restitution_outcome: ImprovementOutcome
    run_level_penetration_outcome: ImprovementOutcome
    primary_penetration_outcome: ImprovementOutcome
    secondary_episode_count: int
    chatter_count: int
    run_level_difference_explained_by_secondary_episodes: bool
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class PrimaryImpactAttributionSummary:
    """Aggregate primary-impact attribution summary."""

    total_cases: int
    primary_scope_cases: int
    fallback_run_level_cases: int
    unavailable_cases: int
    case_outcome_counts: Mapping[str, int]
    restitution_outcome_counts: Mapping[str, int]
    penetration_outcome_counts: Mapping[str, int]
    duration_outcome_counts: Mapping[str, int]
    primary_reason_counts: Mapping[str, int]
    secondary_reason_counts: Mapping[str, int]
    primary_reference_converged_cases: int
    primary_reference_unresolved_cases: int
    matched_primary_cases: int
    unmatched_primary_cases: int
    run_level_unresolved_but_primary_converged_cases: int
    primary_unresolved_but_run_level_converged_cases: int
    mean_primary_restitution_error: float | None
    max_primary_restitution_error: float | None
    mean_primary_penetration_error: float | None
    max_primary_penetration_error: float | None
    mean_primary_duration_error: float | None
    max_primary_duration_error: float | None
    mean_adaptive_step_saving: float | None


@dataclass(frozen=True)
class PrimaryAttributionDataset:
    """Exportable primary attribution dataset."""

    attributions: tuple[PrimaryImpactFailureAttribution, ...]
    differences: tuple[RunPrimaryAttributionDifference, ...]
    summary: PrimaryImpactAttributionSummary
    config: dict[str, object]


def attribute_primary_impact_failure(
    input: PrimaryImpactAttributionInput,
    *,
    improvement_config: PrimaryImpactImprovementConfig = PrimaryImpactImprovementConfig(),
    attribution_config: AdaptiveAttributionConfig = AdaptiveAttributionConfig(),
    prefer_run_level: bool = False,
) -> PrimaryImpactFailureAttribution:
    """Attribute adaptive failure using primary impact whenever possible."""
    primary_status = None if input.primary_reference_convergence is None else input.primary_reference_convergence.overall_status
    run_status = None if input.run_level_reference_convergence is None else input.run_level_reference_convergence.overall_status
    coarse_match_status = None if input.coarse_match is None else input.coarse_match.status
    adaptive_match_status = None if input.adaptive_match is None else input.adaptive_match.status
    evidence: list[str] = []
    secondary: list[AdaptiveFailureReason] = []

    if prefer_run_level:
        return _run_level_attribution(input, run_status, evidence=("explicit run-level attribution requested",))

    if input.adaptive_episode is not None and _invalid_episode(input.adaptive_episode):
        return PrimaryImpactFailureAttribution(
            case_id=input.case_id,
            candidate_id=input.candidate_id,
            scope=AttributionScope.PRIMARY_IMPACT,
            case_outcome=PrimaryImpactCaseOutcome.INVALID_ADAPTIVE,
            primary_reason=AdaptiveFailureReason.NONPHYSICAL_ADAPTIVE_RESULT,
            secondary_reasons=(),
            restitution_outcome=ImprovementOutcome.NOT_APPLICABLE,
            penetration_outcome=ImprovementOutcome.NOT_APPLICABLE,
            duration_outcome=ImprovementOutcome.NOT_APPLICABLE,
            primary_reference_status=primary_status,
            run_level_reference_status=run_status,
            coarse_primary_match_status=coarse_match_status,
            adaptive_primary_match_status=adaptive_match_status,
            evidence=(f"adaptive primary validity = {input.adaptive_episode.validity.value}",),
        )

    if input.reference_episode is None or input.adaptive_episode is None:
        reason = AdaptiveFailureReason.PRIMARY_IMPACT_NOT_FOUND
        evidence.append("primary attribution unavailable because reference or adaptive primary episode is missing")
        fallback = _fallback_or_unavailable(input, reason, secondary, evidence, primary_status, run_status, coarse_match_status, adaptive_match_status)
        return fallback

    if input.adaptive_match is None or input.adaptive_match.status is not EpisodeMatchStatus.MATCHED:
        reason = AdaptiveFailureReason.EPISODE_MISMATCH
        status = "missing" if input.adaptive_match is None else input.adaptive_match.status.value
        evidence.append(f"primary attribution unavailable because adaptive match status = {status}")
        fallback = _fallback_or_unavailable(input, reason, secondary, evidence, primary_status, run_status, coarse_match_status, adaptive_match_status)
        return fallback

    if primary_status is not ReferenceConvergenceStatus.CONVERGED:
        evidence.append(f"primary reference status = {None if primary_status is None else primary_status.value}")
        return PrimaryImpactFailureAttribution(
            case_id=input.case_id,
            candidate_id=input.candidate_id,
            scope=AttributionScope.PRIMARY_IMPACT,
            case_outcome=PrimaryImpactCaseOutcome.REFERENCE_UNRESOLVED,
            primary_reason=AdaptiveFailureReason.REFERENCE_NOT_CONVERGED,
            secondary_reasons=(),
            restitution_outcome=ImprovementOutcome.REFERENCE_UNRESOLVED,
            penetration_outcome=ImprovementOutcome.REFERENCE_UNRESOLVED,
            duration_outcome=ImprovementOutcome.REFERENCE_UNRESOLVED,
            primary_reference_status=primary_status,
            run_level_reference_status=run_status,
            coarse_primary_match_status=coarse_match_status,
            adaptive_primary_match_status=adaptive_match_status,
            evidence=tuple(evidence),
        )

    primary = input.primary_comparison
    restitution = _metric_outcome(
        None if primary is None else primary.coarse_restitution_error,
        None if primary is None else primary.adaptive_restitution_error,
        improvement_config.restitution_error_tolerance,
    )
    penetration = _metric_outcome(
        None if primary is None else primary.coarse_penetration_error,
        None if primary is None else primary.adaptive_penetration_error,
        improvement_config.penetration_error_tolerance,
    )
    duration = _metric_outcome(
        None if primary is None else primary.coarse_duration_error,
        None if primary is None else primary.adaptive_duration_error,
        improvement_config.duration_error_tolerance,
    )
    case_outcome = _case_outcome((restitution, penetration, duration))
    metric_not_improved = any(outcome is ImprovementOutcome.NOT_IMPROVED for outcome in (restitution, penetration, duration))
    if input.adaptive_trace is not None:
        diagnostic = _trace_reasons(input.adaptive_trace, attribution_config, metric_not_improved)
        secondary.extend(diagnostic[0])
        evidence.extend(diagnostic[1])
    if _secondary_count(input) > 0 and case_outcome in {
        PrimaryImpactCaseOutcome.IMPROVED,
        PrimaryImpactCaseOutcome.BOTH_ACCEPTABLE,
    }:
        secondary.append(AdaptiveFailureReason.RUN_LEVEL_SECONDARY_EPISODE_DIFFERENCE)
        evidence.append("primary impact was acceptable; run-level differences may come from later episodes")
    primary_reason = _primary_reason_from_metrics(restitution, penetration, duration, secondary, metric_not_improved)
    return PrimaryImpactFailureAttribution(
        case_id=input.case_id,
        candidate_id=input.candidate_id,
        scope=AttributionScope.PRIMARY_IMPACT,
        case_outcome=case_outcome,
        primary_reason=primary_reason,
        secondary_reasons=tuple(reason for reason in _dedupe(secondary) if reason is not primary_reason),
        restitution_outcome=restitution,
        penetration_outcome=penetration,
        duration_outcome=duration,
        primary_reference_status=primary_status,
        run_level_reference_status=run_status,
        coarse_primary_match_status=coarse_match_status,
        adaptive_primary_match_status=adaptive_match_status,
        evidence=tuple(evidence),
        adaptive_restitution_error=None if primary is None else primary.adaptive_restitution_error,
        adaptive_penetration_error=None if primary is None else primary.adaptive_penetration_error,
        adaptive_duration_error=None if primary is None else primary.adaptive_duration_error,
        adaptive_step_saving=_step_saving(input),
        prediction_lead_macro_steps=_lead(input),
        minimum_timestep=_min_timestep(input),
        maximum_substep_count=_max_substeps(input),
        secondary_episode_count=_secondary_count(input),
    )


def build_run_primary_difference(
    attribution: PrimaryImpactFailureAttribution,
    *,
    run_level_restitution_outcome: ImprovementOutcome,
    run_level_penetration_outcome: ImprovementOutcome,
    secondary_episode_count: int,
    chatter_count: int,
) -> RunPrimaryAttributionDifference:
    """Compare run-level and primary-impact outcomes for one case."""
    explained = (
        secondary_episode_count > 0
        and (
            attribution.restitution_outcome in {ImprovementOutcome.IMPROVED, ImprovementOutcome.BOTH_ACCEPTABLE}
            or attribution.penetration_outcome in {ImprovementOutcome.IMPROVED, ImprovementOutcome.BOTH_ACCEPTABLE}
        )
        and (
            run_level_restitution_outcome is ImprovementOutcome.NOT_IMPROVED
            or run_level_penetration_outcome is ImprovementOutcome.NOT_IMPROVED
        )
    )
    evidence = ()
    if explained:
        evidence = ("primary impact was accurate; run-level discrepancy came from later contact episodes",)
    return RunPrimaryAttributionDifference(
        case_id=attribution.case_id,
        run_level_restitution_outcome=run_level_restitution_outcome,
        primary_restitution_outcome=attribution.restitution_outcome,
        run_level_penetration_outcome=run_level_penetration_outcome,
        primary_penetration_outcome=attribution.penetration_outcome,
        secondary_episode_count=secondary_episode_count,
        chatter_count=chatter_count,
        run_level_difference_explained_by_secondary_episodes=explained,
        evidence=evidence,
    )


def build_primary_attribution_summary(
    attributions: Sequence[PrimaryImpactFailureAttribution],
    *,
    differences: Sequence[RunPrimaryAttributionDifference] = (),
    run_level_comparisons: Sequence[BenchmarkComparison] = (),
) -> PrimaryImpactAttributionSummary:
    """Aggregate primary-impact attribution results."""
    attrs = tuple(attributions)
    primary_errors_r = [attr.adaptive_restitution_error for attr in attrs if attr.adaptive_restitution_error is not None]
    primary_errors_p = [attr.adaptive_penetration_error for attr in attrs if attr.adaptive_penetration_error is not None]
    primary_errors_d = [attr.adaptive_duration_error for attr in attrs if attr.adaptive_duration_error is not None]
    step_savings = [attr.adaptive_step_saving for attr in attrs if attr.adaptive_step_saving is not None]
    secondary = Counter(reason.value for attr in attrs for reason in attr.secondary_reasons)
    return PrimaryImpactAttributionSummary(
        total_cases=len(attrs),
        primary_scope_cases=sum(attr.scope is AttributionScope.PRIMARY_IMPACT for attr in attrs),
        fallback_run_level_cases=sum(attr.scope is AttributionScope.FALLBACK_RUN_LEVEL for attr in attrs),
        unavailable_cases=sum(attr.scope is AttributionScope.UNAVAILABLE for attr in attrs),
        case_outcome_counts=dict(sorted(Counter(attr.case_outcome.value for attr in attrs).items())),
        restitution_outcome_counts=dict(sorted(Counter(attr.restitution_outcome.value for attr in attrs).items())),
        penetration_outcome_counts=dict(sorted(Counter(attr.penetration_outcome.value for attr in attrs).items())),
        duration_outcome_counts=dict(sorted(Counter(attr.duration_outcome.value for attr in attrs).items())),
        primary_reason_counts=dict(sorted(Counter(attr.primary_reason.value for attr in attrs).items())),
        secondary_reason_counts=dict(sorted(secondary.items())),
        primary_reference_converged_cases=sum(attr.primary_reference_status is ReferenceConvergenceStatus.CONVERGED for attr in attrs),
        primary_reference_unresolved_cases=sum(
            attr.primary_reference_status not in (None, ReferenceConvergenceStatus.CONVERGED) for attr in attrs
        ),
        matched_primary_cases=sum(attr.adaptive_primary_match_status is EpisodeMatchStatus.MATCHED for attr in attrs),
        unmatched_primary_cases=sum(attr.adaptive_primary_match_status not in (None, EpisodeMatchStatus.MATCHED) for attr in attrs),
        run_level_unresolved_but_primary_converged_cases=sum(
            attr.run_level_reference_status not in (None, ReferenceConvergenceStatus.CONVERGED)
            and attr.primary_reference_status is ReferenceConvergenceStatus.CONVERGED
            for attr in attrs
        ),
        primary_unresolved_but_run_level_converged_cases=sum(
            attr.primary_reference_status not in (None, ReferenceConvergenceStatus.CONVERGED)
            and attr.run_level_reference_status is ReferenceConvergenceStatus.CONVERGED
            for attr in attrs
        ),
        mean_primary_restitution_error=_mean(primary_errors_r),
        max_primary_restitution_error=None if not primary_errors_r else max(primary_errors_r),
        mean_primary_penetration_error=_mean(primary_errors_p),
        max_primary_penetration_error=None if not primary_errors_p else max(primary_errors_p),
        mean_primary_duration_error=_mean(primary_errors_d),
        max_primary_duration_error=None if not primary_errors_d else max(primary_errors_d),
        mean_adaptive_step_saving=_mean(step_savings),
    )


def primary_improvement_rates(attributions: Sequence[PrimaryImpactFailureAttribution]) -> dict[str, float]:
    """Compute primary-impact improvement rates with filtered denominators."""
    return {
        "restitution": _rate(attr.restitution_outcome for attr in attributions),
        "penetration": _rate(attr.penetration_outcome for attr in attributions),
        "duration": _rate(attr.duration_outcome for attr in attributions),
        "case": _case_rate(attr.case_outcome for attr in attributions),
    }


def export_primary_attribution_csv(attributions: Sequence[PrimaryImpactFailureAttribution], path: str | Path) -> None:
    """Export primary attribution CSV."""
    rows = []
    for attr in attributions:
        rows.append({
            "case_id": attr.case_id,
            "candidate_id": attr.candidate_id,
            "scope": attr.scope.value,
            "case_outcome": attr.case_outcome.value,
            "primary_reason": attr.primary_reason.value,
            "secondary_reasons": ";".join(reason.value for reason in attr.secondary_reasons),
            "primary_reference_status": None if attr.primary_reference_status is None else attr.primary_reference_status.value,
            "coarse_match_status": None if attr.coarse_primary_match_status is None else attr.coarse_primary_match_status.value,
            "adaptive_match_status": None if attr.adaptive_primary_match_status is None else attr.adaptive_primary_match_status.value,
            "restitution_outcome": attr.restitution_outcome.value,
            "penetration_outcome": attr.penetration_outcome.value,
            "duration_outcome": attr.duration_outcome.value,
            "restitution_error": attr.adaptive_restitution_error,
            "penetration_error": attr.adaptive_penetration_error,
            "duration_error": attr.adaptive_duration_error,
            "prediction_lead": attr.prediction_lead_macro_steps,
            "minimum_timestep": attr.minimum_timestep,
            "maximum_substep_count": attr.maximum_substep_count,
            "step_saving": attr.adaptive_step_saving,
            "secondary_episode_count": attr.secondary_episode_count,
            "chatter_count": attr.chatter_count,
            "evidence": " | ".join(attr.evidence),
        })
    _write_csv(rows, path)


def export_run_primary_difference_csv(differences: Sequence[RunPrimaryAttributionDifference], path: str | Path) -> None:
    """Export run-vs-primary attribution differences."""
    _write_csv([
        {
            "case_id": item.case_id,
            "run_level_restitution_outcome": item.run_level_restitution_outcome.value,
            "primary_restitution_outcome": item.primary_restitution_outcome.value,
            "run_level_penetration_outcome": item.run_level_penetration_outcome.value,
            "primary_penetration_outcome": item.primary_penetration_outcome.value,
            "secondary_episode_count": item.secondary_episode_count,
            "chatter_count": item.chatter_count,
            "explained_by_secondary": item.run_level_difference_explained_by_secondary_episodes,
            "evidence": " | ".join(item.evidence),
        }
        for item in differences
    ], path)


def export_primary_attribution_json(dataset: PrimaryAttributionDataset, path: str | Path) -> None:
    """Export primary attribution diagnostics JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_dataset_to_dict(dataset), indent=2, sort_keys=True), encoding="utf8")


def write_primary_attribution_markdown_report(dataset: PrimaryAttributionDataset, path: str | Path) -> None:
    """Write primary attribution Markdown report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_primary_attribution_markdown_report(dataset), encoding="utf8")


def build_primary_attribution_markdown_report(dataset: PrimaryAttributionDataset) -> str:
    """Build a compact primary attribution Markdown report."""
    summary = dataset.summary
    lines = [
        "# Primary Impact Attribution",
        "",
        "## Overview",
        "",
        f"- total cases: {summary.total_cases}",
        f"- primary / fallback / unavailable: {summary.primary_scope_cases} / {summary.fallback_run_level_cases} / {summary.unavailable_cases}",
        f"- primary matched: {summary.matched_primary_cases}",
        f"- primary reference converged: {summary.primary_reference_converged_cases}",
        "",
        "## Primary-Impact Outcomes",
        "",
        f"- case outcomes: {dict(summary.case_outcome_counts)}",
        f"- restitution outcomes: {dict(summary.restitution_outcome_counts)}",
        f"- penetration outcomes: {dict(summary.penetration_outcome_counts)}",
        f"- duration outcomes: {dict(summary.duration_outcome_counts)}",
        "",
        "## Run-Level Versus Primary-Impact",
        "",
        f"- run-level unresolved but primary converged: {summary.run_level_unresolved_but_primary_converged_cases}",
        f"- primary unresolved but run-level converged: {summary.primary_unresolved_but_run_level_converged_cases}",
        "",
        "## Failure Reasons",
        "",
    ]
    for reason, count in summary.primary_reason_counts.items():
        lines.append(f"- {reason}: {count}")
    contaminated = [item for item in dataset.differences if item.run_level_difference_explained_by_secondary_episodes]
    lines.extend(["", "## Secondary Episode Contamination", ""])
    lines.extend([f"- {item.case_id}: {item.evidence[0]}" for item in contaminated] or ["- none"])
    lines.extend(["", "## Conclusion", "", _conclusion(summary)])
    return "\n".join(lines) + "\n"


def _fallback_or_unavailable(
    input: PrimaryImpactAttributionInput,
    reason: AdaptiveFailureReason,
    secondary: list[AdaptiveFailureReason],
    evidence: list[str],
    primary_status: ReferenceConvergenceStatus | None,
    run_status: ReferenceConvergenceStatus | None,
    coarse_match_status: EpisodeMatchStatus | None,
    adaptive_match_status: EpisodeMatchStatus | None,
) -> PrimaryImpactFailureAttribution:
    if input.run_level_comparison is not None:
        secondary.append(reason)
        evidence.append("falling back to run-level comparison")
        run = _run_level_attribution(input, run_status, evidence=tuple(evidence))
        return PrimaryImpactFailureAttribution(
            **{
                **run.__dict__,
                "scope": AttributionScope.FALLBACK_RUN_LEVEL,
                "secondary_reasons": tuple(_dedupe((*run.secondary_reasons, *secondary))),
                "primary_reference_status": primary_status,
                "coarse_primary_match_status": coarse_match_status,
                "adaptive_primary_match_status": adaptive_match_status,
            }
        )
    return PrimaryImpactFailureAttribution(
        case_id=input.case_id,
        candidate_id=input.candidate_id,
        scope=AttributionScope.UNAVAILABLE,
        case_outcome=PrimaryImpactCaseOutcome.EPISODE_UNMATCHED,
        primary_reason=reason,
        secondary_reasons=tuple(_dedupe(secondary)),
        restitution_outcome=ImprovementOutcome.EPISODE_UNMATCHED,
        penetration_outcome=ImprovementOutcome.EPISODE_UNMATCHED,
        duration_outcome=ImprovementOutcome.EPISODE_UNMATCHED,
        primary_reference_status=primary_status,
        run_level_reference_status=run_status,
        coarse_primary_match_status=coarse_match_status,
        adaptive_primary_match_status=adaptive_match_status,
        evidence=tuple(evidence),
    )


def _run_level_attribution(
    input: PrimaryImpactAttributionInput,
    run_status: ReferenceConvergenceStatus | None,
    *,
    evidence: tuple[str, ...],
) -> PrimaryImpactFailureAttribution:
    comparison = input.run_level_comparison
    restitution = ImprovementOutcome.NOT_APPLICABLE if comparison is None else _metric_outcome(
        comparison.coarse_restitution_error,
        comparison.adaptive_restitution_error,
        0.005,
    )
    penetration = ImprovementOutcome.NOT_APPLICABLE if comparison is None else _metric_outcome(
        comparison.coarse_penetration_error,
        comparison.adaptive_penetration_error,
        5.0e-4,
    )
    case_outcome = _case_outcome((restitution, penetration))
    return PrimaryImpactFailureAttribution(
        case_id=input.case_id,
        candidate_id=input.candidate_id,
        scope=AttributionScope.RUN_LEVEL,
        case_outcome=case_outcome,
        primary_reason=AdaptiveFailureReason.NONE if case_outcome in {PrimaryImpactCaseOutcome.IMPROVED, PrimaryImpactCaseOutcome.BOTH_ACCEPTABLE} else AdaptiveFailureReason.UNKNOWN,
        secondary_reasons=(),
        restitution_outcome=restitution,
        penetration_outcome=penetration,
        duration_outcome=ImprovementOutcome.NOT_APPLICABLE,
        primary_reference_status=None,
        run_level_reference_status=run_status,
        coarse_primary_match_status=None,
        adaptive_primary_match_status=None,
        evidence=evidence,
    )


def _metric_outcome(coarse_error: float | None, adaptive_error: float | None, tolerance: float) -> ImprovementOutcome:
    if coarse_error is None or adaptive_error is None:
        return ImprovementOutcome.NOT_APPLICABLE
    if coarse_error <= tolerance and adaptive_error <= tolerance:
        return ImprovementOutcome.BOTH_ACCEPTABLE
    if adaptive_error + tolerance < coarse_error:
        return ImprovementOutcome.IMPROVED
    if adaptive_error > coarse_error + tolerance:
        return ImprovementOutcome.NOT_IMPROVED
    return ImprovementOutcome.BOTH_ACCEPTABLE


def _case_outcome(outcomes: Sequence[ImprovementOutcome]) -> PrimaryImpactCaseOutcome:
    applicable = [item for item in outcomes if item is not ImprovementOutcome.NOT_APPLICABLE]
    if not applicable:
        return PrimaryImpactCaseOutcome.BOTH_ACCEPTABLE
    if any(item is ImprovementOutcome.REFERENCE_UNRESOLVED for item in applicable):
        return PrimaryImpactCaseOutcome.REFERENCE_UNRESOLVED
    if any(item is ImprovementOutcome.EPISODE_UNMATCHED for item in applicable):
        return PrimaryImpactCaseOutcome.EPISODE_UNMATCHED
    improved = any(item is ImprovementOutcome.IMPROVED for item in applicable)
    not_improved = any(item is ImprovementOutcome.NOT_IMPROVED for item in applicable)
    if improved and not_improved:
        return PrimaryImpactCaseOutcome.PARTIALLY_IMPROVED
    if improved:
        return PrimaryImpactCaseOutcome.IMPROVED
    if not_improved:
        return PrimaryImpactCaseOutcome.NOT_IMPROVED
    return PrimaryImpactCaseOutcome.BOTH_ACCEPTABLE


def _primary_reason_from_metrics(
    restitution: ImprovementOutcome,
    penetration: ImprovementOutcome,
    duration: ImprovementOutcome,
    secondary: Sequence[AdaptiveFailureReason],
    metric_not_improved: bool,
) -> AdaptiveFailureReason:
    if metric_not_improved:
        for reason in (
            AdaptiveFailureReason.LATE_PREDICTION,
            AdaptiveFailureReason.SHORT_PREDICTION_LEAD,
            AdaptiveFailureReason.MAX_SUBSTEPS_LIMITED,
            AdaptiveFailureReason.INSUFFICIENT_TIME_RESOLUTION,
            AdaptiveFailureReason.EARLY_FINE_EXIT,
        ):
            if reason in secondary:
                return reason
        if restitution is ImprovementOutcome.NOT_IMPROVED:
            return AdaptiveFailureReason.PRIMARY_RESTITUTION_NOT_IMPROVED
        if penetration is ImprovementOutcome.NOT_IMPROVED:
            return AdaptiveFailureReason.PRIMARY_PENETRATION_NOT_IMPROVED
        if duration is ImprovementOutcome.NOT_IMPROVED:
            return AdaptiveFailureReason.PRIMARY_DURATION_NOT_IMPROVED
    return AdaptiveFailureReason.NONE


def _trace_reasons(
    trace: AdaptiveDiagnosticTrace,
    config: AdaptiveAttributionConfig,
    metric_not_improved: bool,
) -> tuple[list[AdaptiveFailureReason], list[str]]:
    reasons: list[AdaptiveFailureReason] = []
    evidence: list[str] = []
    if not trace.episodes:
        return reasons, evidence
    episode = trace.episodes[0]
    lead = episode.prediction_lead_time_macro_steps
    if lead is not None:
        evidence.append(f"prediction lead = {lead:.3g} macro steps")
        if metric_not_improved and lead <= 0.0:
            reasons.append(AdaptiveFailureReason.LATE_PREDICTION)
        elif metric_not_improved and lead < config.short_prediction_lead_macro_steps:
            reasons.append(AdaptiveFailureReason.SHORT_PREDICTION_LEAD)
    if episode.limited_by_maximum_substeps:
        evidence.append("maximum_substeps was reached")
        if metric_not_improved:
            reasons.append(AdaptiveFailureReason.MAX_SUBSTEPS_LIMITED)
    if episode.solver_characteristic_timescale is not None:
        samples = episode.solver_characteristic_timescale / episode.minimum_actual_timestep
        evidence.append(f"samples per solver timescale = {samples:.3g}")
        if metric_not_improved and samples < config.samples_per_characteristic_time and not episode.limited_by_maximum_substeps:
            reasons.append(AdaptiveFailureReason.INSUFFICIENT_TIME_RESOLUTION)
    return reasons, evidence


def _invalid_episode(episode: ContactEpisodeMetrics) -> bool:
    return (
        episode.validity.value in {"unstable", "nonphysical_rebound"}
        or _bad(episode.restitution)
        or _bad(episode.maximum_penetration)
    )


def _secondary_count(input: PrimaryImpactAttributionInput) -> int:
    if input.adaptive_trace is None:
        return 0
    return max(0, input.adaptive_trace.total_contact_episode_count - 1)


def _lead(input: PrimaryImpactAttributionInput) -> float | None:
    if input.adaptive_trace is None or not input.adaptive_trace.episodes:
        return None
    return input.adaptive_trace.episodes[0].prediction_lead_time_macro_steps


def _min_timestep(input: PrimaryImpactAttributionInput) -> float | None:
    return None if input.adaptive_trace is None else input.adaptive_trace.global_minimum_timestep


def _max_substeps(input: PrimaryImpactAttributionInput) -> int | None:
    return None if input.adaptive_trace is None else input.adaptive_trace.global_maximum_substep_count


def _step_saving(input: PrimaryImpactAttributionInput) -> float | None:
    return None if input.run_level_comparison is None else input.run_level_comparison.adaptive_step_saving


def _rate(outcomes) -> float:
    values = [item for item in outcomes if item not in {
        ImprovementOutcome.REFERENCE_UNRESOLVED,
        ImprovementOutcome.EPISODE_UNMATCHED,
        ImprovementOutcome.NOT_APPLICABLE,
    }]
    return 0.0 if not values else sum(item in {ImprovementOutcome.IMPROVED, ImprovementOutcome.BOTH_ACCEPTABLE} for item in values) / len(values)


def _case_rate(outcomes) -> float:
    values = [item for item in outcomes if item not in {
        PrimaryImpactCaseOutcome.REFERENCE_UNRESOLVED,
        PrimaryImpactCaseOutcome.EPISODE_UNMATCHED,
        PrimaryImpactCaseOutcome.INVALID_ADAPTIVE,
    }]
    return 0.0 if not values else sum(item in {PrimaryImpactCaseOutcome.IMPROVED, PrimaryImpactCaseOutcome.BOTH_ACCEPTABLE, PrimaryImpactCaseOutcome.PARTIALLY_IMPROVED} for item in values) / len(values)


def _comparison_for(case_id: str, comparisons: Sequence[BenchmarkComparison]) -> BenchmarkComparison | None:
    return next((item for item in comparisons if item.case_id == case_id), None)


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _bad(value: float | None) -> bool:
    return value is not None and not math.isfinite(value)


def _dedupe(values: Sequence[AdaptiveFailureReason]) -> list[AdaptiveFailureReason]:
    result: list[AdaptiveFailureReason] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _write_csv(rows: Sequence[dict[str, object]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_to_dict(dataset: PrimaryAttributionDataset) -> dict[str, object]:
    return {
        "attributions": [_attr_to_dict(item) for item in dataset.attributions],
        "differences": [_difference_to_dict(item) for item in dataset.differences],
        "summary": asdict(dataset.summary),
        "config": dataset.config,
    }


def _attr_to_dict(attr: PrimaryImpactFailureAttribution) -> dict[str, object]:
    data = asdict(attr)
    data["scope"] = attr.scope.value
    data["case_outcome"] = attr.case_outcome.value
    data["primary_reason"] = attr.primary_reason.value
    data["secondary_reasons"] = [reason.value for reason in attr.secondary_reasons]
    data["restitution_outcome"] = attr.restitution_outcome.value
    data["penetration_outcome"] = attr.penetration_outcome.value
    data["duration_outcome"] = attr.duration_outcome.value
    data["primary_reference_status"] = None if attr.primary_reference_status is None else attr.primary_reference_status.value
    data["run_level_reference_status"] = None if attr.run_level_reference_status is None else attr.run_level_reference_status.value
    data["coarse_primary_match_status"] = None if attr.coarse_primary_match_status is None else attr.coarse_primary_match_status.value
    data["adaptive_primary_match_status"] = None if attr.adaptive_primary_match_status is None else attr.adaptive_primary_match_status.value
    return data


def _difference_to_dict(item: RunPrimaryAttributionDifference) -> dict[str, object]:
    data = asdict(item)
    data["run_level_restitution_outcome"] = item.run_level_restitution_outcome.value
    data["primary_restitution_outcome"] = item.primary_restitution_outcome.value
    data["run_level_penetration_outcome"] = item.run_level_penetration_outcome.value
    data["primary_penetration_outcome"] = item.primary_penetration_outcome.value
    return data


def _conclusion(summary: PrimaryImpactAttributionSummary) -> str:
    if summary.run_level_unresolved_but_primary_converged_cases:
        return "Some run-level unresolved cases had converged primary impacts."
    if summary.primary_scope_cases == summary.total_cases:
        return "All selected cases were attributed at primary-impact scope."
    return "Primary-impact attribution was unavailable for at least one case."
