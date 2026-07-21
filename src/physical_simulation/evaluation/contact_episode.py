"""Episode-level contact segmentation, matching, and diagnostics."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import contact_benchmark as _benchmark
from physical_simulation.evaluation.contact_benchmark import (
    BenchmarkMode,
    BenchmarkValidity,
    ContactBenchmarkCase,
    SpherePlaneBenchmarkCase,
    SphereSphereBenchmarkCase,
    classify_benchmark_validity,
)
from physical_simulation.evaluation.contact_calibration import RestitutionOutcome
from physical_simulation.evaluation.contact_convergence import (
    ReferenceConvergenceConfig,
    ReferenceConvergenceStatus,
    ReferenceMetricConvergence,
    _metric_convergence,
)
from physical_simulation.mujoco import (
    AdaptiveMuJoCoRunner,
    AdaptiveSubstepConfig,
    ContactMotionState,
    SubstepRecommendationConfig,
)
from physical_simulation.runtime import SimulationStepResult
from physical_simulation.validation.asset_validator import _finite_float
from physical_simulation.validation.errors import PhysicsValidationError

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class ContactEpisodeSample:
    """One contact-candidate sample at an actual physics step."""

    simulation_time: float
    physics_step_index: int
    macro_step_index: int | None
    candidate_id: str
    body_a_id: str
    body_b_id: str
    active_contact: bool
    gap_or_distance: float | None
    penetration: float
    contact_normal: Vector3 | None
    normal_relative_velocity: float | None
    body_a_linear_velocity: Vector3
    body_b_linear_velocity: Vector3
    body_a_angular_velocity: Vector3
    body_b_angular_velocity: Vector3
    substep_count: int
    timestep: float
    adaptive_state: ContactMotionState | None


@dataclass(frozen=True)
class RawContactInterval:
    """A contiguous interval of active contact samples."""

    interval_index: int
    candidate_id: str
    body_a_id: str
    body_b_id: str
    start_sample_index: int
    end_sample_index: int
    start_time: float
    end_time: float
    duration_seconds: float
    samples: tuple[ContactEpisodeSample, ...]
    ended_while_contact_active: bool = False


class IntervalMergeReason(Enum):
    """Reason an adjacent raw interval was merged into an episode."""

    NONE = "none"
    SHORT_GAP = "short_gap"
    INSUFFICIENT_SEPARATION = "insufficient_separation"
    CONTACT_CHATTER = "contact_chatter"


class ContactEpisodeKind(Enum):
    """Semantic class for a segmented contact episode."""

    PRIMARY_IMPACT = "primary_impact"
    SECONDARY_IMPACT = "secondary_impact"
    RESTING_CONTACT = "resting_contact"
    CONTACT_CHATTER = "contact_chatter"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class ContactEpisodeSegmentationConfig:
    """Configuration for interval merging and episode classification."""

    maximum_chatter_gap_seconds: float
    maximum_chatter_gap_steps: int = 2
    minimum_separation_speed: float = 0.02
    minimum_separation_duration: float = 0.0
    minimum_episode_duration: float = 0.0
    minimum_impact_speed: float = 0.02
    resting_speed_threshold: float = 0.01
    resting_duration_threshold: float = 0.02

    def __post_init__(self) -> None:
        for field_name in (
            "maximum_chatter_gap_seconds",
            "minimum_separation_speed",
            "minimum_separation_duration",
            "minimum_episode_duration",
            "minimum_impact_speed",
            "resting_speed_threshold",
            "resting_duration_threshold",
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
        if (
            not isinstance(self.maximum_chatter_gap_steps, int)
            or isinstance(self.maximum_chatter_gap_steps, bool)
            or self.maximum_chatter_gap_steps < 0
        ):
            raise PhysicsValidationError("maximum_chatter_gap_steps must be a non-negative int")


@dataclass(frozen=True)
class ContactEpisodeMetrics:
    """Metrics for one segmented contact episode."""

    episode_index: int
    candidate_id: str
    kind: ContactEpisodeKind
    body_a_id: str
    body_b_id: str
    start_time: float
    end_time: float
    duration_seconds: float
    raw_interval_count: int
    merged_gap_count: int
    pre_contact_normal_velocity: float | None
    impact_speed: float | None
    post_contact_normal_velocity: float | None
    separation_speed: float | None
    restitution: float | None
    maximum_penetration: float
    normalized_penetration: float | None
    maximum_penetration_time: float | None
    contact_sample_count: int
    physics_step_count: int
    minimum_timestep: float
    maximum_timestep: float
    minimum_substep_count: int
    maximum_substep_count: int
    state_at_start: ContactMotionState | None
    state_at_maximum_penetration: ContactMotionState | None
    state_at_end: ContactMotionState | None
    ended_with_separation: bool
    ended_while_contact_active: bool
    validity: BenchmarkValidity


@dataclass(frozen=True)
class ContactEpisodeStatistics:
    """Run-level episode segmentation statistics."""

    raw_interval_count: int
    merged_episode_count: int
    primary_impact_count: int
    secondary_impact_count: int
    resting_contact_count: int
    chatter_interval_count: int
    merged_chatter_gap_count: int
    unmatched_episode_count: int


class EpisodeMatchStatus(Enum):
    """Status for matching an episode across two modes."""

    MATCHED = "matched"
    UNMATCHED_REFERENCE = "unmatched_reference"
    UNMATCHED_CANDIDATE = "unmatched_candidate"
    AMBIGUOUS = "ambiguous"
    INCOMPATIBLE_KIND = "incompatible_kind"


@dataclass(frozen=True)
class EpisodeMatch:
    """One reference/comparison episode match."""

    candidate_id: str
    reference_episode_index: int | None
    comparison_episode_index: int | None
    status: EpisodeMatchStatus
    reference_kind: ContactEpisodeKind | None
    comparison_kind: ContactEpisodeKind | None
    start_time_difference: float | None
    impact_speed_difference: float | None
    reason: str


@dataclass(frozen=True)
class EpisodeMatchingConfig:
    """Configuration for deterministic episode matching."""

    maximum_start_time_difference: float
    maximum_relative_impact_speed_difference: float = 0.25
    require_same_candidate: bool = True
    require_compatible_kind: bool = True


@dataclass(frozen=True)
class EpisodeMetricComparison:
    """Metric errors for a matched episode pair."""

    case_id: str
    candidate_id: str
    reference_episode_index: int
    comparison_episode_index: int
    reference_mode: BenchmarkMode
    comparison_mode: BenchmarkMode
    kind: ContactEpisodeKind
    restitution_error: float | None
    penetration_error: float
    duration_error: float | None
    impact_speed_error: float | None
    separation_speed_error: float | None
    start_time_error: float
    maximum_penetration_time_error: float | None
    reference_validity: BenchmarkValidity
    comparison_validity: BenchmarkValidity


@dataclass(frozen=True)
class PrimaryImpactBenchmarkComparison:
    """Primary impact comparison for coarse/adaptive against a reference."""

    case_id: str
    candidate_id: str
    reference_episode: ContactEpisodeMetrics
    coarse_episode: ContactEpisodeMetrics | None
    adaptive_episode: ContactEpisodeMetrics | None
    coarse_match_status: EpisodeMatchStatus
    adaptive_match_status: EpisodeMatchStatus
    coarse_restitution_error: float | None
    adaptive_restitution_error: float | None
    coarse_penetration_error: float | None
    adaptive_penetration_error: float | None
    coarse_duration_error: float | None
    adaptive_duration_error: float | None
    adaptive_improves_restitution: bool | None
    adaptive_improves_penetration: bool | None


@dataclass(frozen=True)
class EpisodeReferenceLevelResult:
    """Reference refinement level for one matched episode."""

    refinement_level: int
    timestep: float
    episode_index: int
    episode_kind: ContactEpisodeKind
    start_time: float
    restitution: float | None
    maximum_penetration: float
    contact_duration_seconds: float | None
    match_status: EpisodeMatchStatus


@dataclass(frozen=True)
class EpisodeReferenceConvergenceResult:
    """Episode-level reference convergence result."""

    case_id: str
    candidate_id: str
    episode_kind: ContactEpisodeKind
    levels: tuple[EpisodeReferenceLevelResult, ...]
    restitution: ReferenceMetricConvergence
    maximum_penetration: ReferenceMetricConvergence
    contact_duration: ReferenceMetricConvergence
    start_time: ReferenceMetricConvergence
    overall_status: ReferenceConvergenceStatus


@dataclass(frozen=True)
class ContactEpisodeAnalysisDataset:
    """Exportable episode analysis dataset."""

    case_id: str
    episodes_by_mode: dict[str, tuple[ContactEpisodeMetrics, ...]]
    matches: tuple[EpisodeMatch, ...]
    comparisons: tuple[EpisodeMetricComparison, ...]
    primary_comparison: PrimaryImpactBenchmarkComparison | None
    reference_convergence: EpisodeReferenceConvergenceResult | None
    statistics_by_mode: dict[str, ContactEpisodeStatistics]


def extract_raw_contact_intervals(
    samples: Sequence[ContactEpisodeSample],
) -> tuple[RawContactInterval, ...]:
    """Extract deterministic contiguous active-contact intervals from samples."""
    intervals: list[RawContactInterval] = []
    start: int | None = None
    for index, sample in enumerate(samples):
        if sample.active_contact and start is None:
            start = index
        if (not sample.active_contact or index == len(samples) - 1) and start is not None:
            end = index if sample.active_contact and index == len(samples) - 1 else index - 1
            chunk = tuple(samples[start : end + 1])
            intervals.append(
                RawContactInterval(
                    interval_index=len(intervals),
                    candidate_id=chunk[0].candidate_id,
                    body_a_id=chunk[0].body_a_id,
                    body_b_id=chunk[0].body_b_id,
                    start_sample_index=start,
                    end_sample_index=end,
                    start_time=chunk[0].simulation_time,
                    end_time=chunk[-1].simulation_time,
                    duration_seconds=max(0.0, chunk[-1].simulation_time - chunk[0].simulation_time),
                    samples=chunk,
                    ended_while_contact_active=sample.active_contact and index == len(samples) - 1,
                )
            )
            start = None
    return tuple(intervals)


def segment_contact_episodes(
    samples: Sequence[ContactEpisodeSample],
    *,
    config: ContactEpisodeSegmentationConfig,
    validation_maximum_restitution: float = 1.05,
    validation_maximum_normalized_penetration: float = 0.25,
) -> tuple[ContactEpisodeMetrics, ...]:
    """Merge raw intervals into physical episodes and compute episode metrics."""
    ordered = tuple(sorted(samples, key=lambda item: (item.candidate_id, item.simulation_time, item.physics_step_index)))
    intervals = extract_raw_contact_intervals(ordered)
    grouped: dict[str, list[RawContactInterval]] = {}
    for interval in intervals:
        grouped.setdefault(interval.candidate_id, []).append(interval)
    episodes: list[ContactEpisodeMetrics] = []
    for candidate_id in sorted(grouped):
        current: list[RawContactInterval] = []
        merged_reasons: list[IntervalMergeReason] = []
        intervals_for_candidate = grouped[candidate_id]
        for interval in intervals_for_candidate:
            if not current:
                current = [interval]
                continue
            reason = _merge_reason(ordered, current[-1], interval, config)
            if reason is not IntervalMergeReason.NONE:
                current.append(interval)
                merged_reasons.append(reason)
            else:
                episodes.append(_build_episode(ordered, current, merged_reasons, config, validation_maximum_restitution, validation_maximum_normalized_penetration))
                current = [interval]
                merged_reasons = []
        if current:
            episodes.append(_build_episode(ordered, current, merged_reasons, config, validation_maximum_restitution, validation_maximum_normalized_penetration))
    episodes = _classify_episodes(episodes, config)
    return tuple(sorted(episodes, key=lambda item: (item.candidate_id, item.start_time, item.episode_index)))


def collect_contact_episode_samples(
    case: ContactBenchmarkCase,
    *,
    mode: BenchmarkMode,
    recommendation: SubstepRecommendationConfig = SubstepRecommendationConfig(maximum_substeps=16),
) -> tuple[ContactEpisodeSample, ...]:
    """Collect candidate samples for fixed coarse, fixed fine, or adaptive mode."""
    macro_steps = _benchmark._macro_steps(case)
    fine_substeps = recommendation.maximum_substeps
    timestep = case.macro_timestep if mode is not BenchmarkMode.FIXED_FINE else case.macro_timestep / fine_substeps
    backend = MuJoCoBackend()
    try:
        backend.load_scene(_benchmark._scene_for_case(case, timestep=timestep))
        _benchmark._apply_initial_velocity(case, backend, update_initial=True)
        if mode is BenchmarkMode.ADAPTIVE:
            return _collect_adaptive_samples(case, backend, recommendation, macro_steps)
        steps = macro_steps if mode is BenchmarkMode.FIXED_COARSE else macro_steps * fine_substeps
        result = backend.reset()
        samples = [_sample_from_result(case, result, macro_step_index=0, substep_count=1, timestep=timestep, adaptive_state=None)]
        for step in range(steps):
            result = backend.step()
            macro_index = step + 1 if mode is BenchmarkMode.FIXED_COARSE else (step + 1) // fine_substeps
            samples.append(_sample_from_result(case, result, macro_step_index=macro_index, substep_count=1, timestep=timestep, adaptive_state=None))
        return tuple(samples)
    finally:
        backend.close()


def match_contact_episodes(
    *,
    reference: Sequence[ContactEpisodeMetrics],
    comparison: Sequence[ContactEpisodeMetrics],
    config: EpisodeMatchingConfig,
) -> tuple[EpisodeMatch, ...]:
    """Deterministically match reference and comparison episodes."""
    matches: list[EpisodeMatch] = []
    used: set[int] = set()
    for ref in sorted(reference, key=lambda item: (item.candidate_id, _kind_rank(item.kind), item.start_time, item.episode_index)):
        candidates = []
        for comp in comparison:
            if comp.episode_index in used:
                continue
            if config.require_same_candidate and comp.candidate_id != ref.candidate_id:
                continue
            if config.require_compatible_kind and not _compatible_kind(ref.kind, comp.kind):
                continue
            start_diff = abs(ref.start_time - comp.start_time)
            if start_diff > config.maximum_start_time_difference:
                continue
            impact_diff = _optional_abs(ref.impact_speed, comp.impact_speed)
            if impact_diff is not None and ref.impact_speed not in (None, 0.0):
                if impact_diff / max(abs(ref.impact_speed), 1.0e-12) > config.maximum_relative_impact_speed_difference:
                    continue
            candidates.append((start_diff, impact_diff, comp))
        if not candidates:
            matches.append(_unmatched_reference(ref, "no comparison episode passed matching thresholds"))
            continue
        candidates.sort(key=lambda item: (item[0], math.inf if item[1] is None else item[1], item[2].episode_index))
        if len(candidates) > 1 and math.isclose(candidates[0][0], candidates[1][0], abs_tol=1.0e-12):
            matches.append(EpisodeMatch(ref.candidate_id, ref.episode_index, None, EpisodeMatchStatus.AMBIGUOUS, ref.kind, None, None, None, "multiple comparison episodes matched equally"))
            continue
        comp = candidates[0][2]
        used.add(comp.episode_index)
        matches.append(
            EpisodeMatch(
                candidate_id=ref.candidate_id,
                reference_episode_index=ref.episode_index,
                comparison_episode_index=comp.episode_index,
                status=EpisodeMatchStatus.MATCHED,
                reference_kind=ref.kind,
                comparison_kind=comp.kind,
                start_time_difference=abs(ref.start_time - comp.start_time),
                impact_speed_difference=_optional_abs(ref.impact_speed, comp.impact_speed),
                reason="matched by candidate, kind, start time, and impact speed",
            )
        )
    for comp in sorted(comparison, key=lambda item: (item.candidate_id, item.start_time, item.episode_index)):
        if comp.episode_index not in used:
            matches.append(
                EpisodeMatch(comp.candidate_id, None, comp.episode_index, EpisodeMatchStatus.UNMATCHED_CANDIDATE, None, comp.kind, None, None, "comparison episode had no reference match")
            )
    return tuple(matches)


def compare_matched_episodes(
    *,
    case_id: str,
    reference_mode: BenchmarkMode,
    comparison_mode: BenchmarkMode,
    reference: Sequence[ContactEpisodeMetrics],
    comparison: Sequence[ContactEpisodeMetrics],
    matches: Sequence[EpisodeMatch],
) -> tuple[EpisodeMetricComparison, ...]:
    """Compute errors for matched episode pairs."""
    ref_by_index = {item.episode_index: item for item in reference}
    comp_by_index = {item.episode_index: item for item in comparison}
    rows: list[EpisodeMetricComparison] = []
    for match in matches:
        if match.status is not EpisodeMatchStatus.MATCHED:
            continue
        ref = ref_by_index[match.reference_episode_index]  # type: ignore[index]
        comp = comp_by_index[match.comparison_episode_index]  # type: ignore[index]
        rows.append(
            EpisodeMetricComparison(
                case_id=case_id,
                candidate_id=ref.candidate_id,
                reference_episode_index=ref.episode_index,
                comparison_episode_index=comp.episode_index,
                reference_mode=reference_mode,
                comparison_mode=comparison_mode,
                kind=ref.kind,
                restitution_error=_optional_abs(ref.restitution, comp.restitution),
                penetration_error=abs(ref.maximum_penetration - comp.maximum_penetration),
                duration_error=_optional_abs(ref.duration_seconds, comp.duration_seconds),
                impact_speed_error=_optional_abs(ref.impact_speed, comp.impact_speed),
                separation_speed_error=_optional_abs(ref.separation_speed, comp.separation_speed),
                start_time_error=abs(ref.start_time - comp.start_time),
                maximum_penetration_time_error=_optional_abs(ref.maximum_penetration_time, comp.maximum_penetration_time),
                reference_validity=ref.validity,
                comparison_validity=comp.validity,
            )
        )
    return tuple(rows)


def build_primary_impact_comparison(
    *,
    case_id: str,
    reference: Sequence[ContactEpisodeMetrics],
    coarse: Sequence[ContactEpisodeMetrics],
    adaptive: Sequence[ContactEpisodeMetrics],
    matching: EpisodeMatchingConfig,
) -> PrimaryImpactBenchmarkComparison | None:
    """Build primary-impact coarse/adaptive comparison against reference."""
    ref = _primary(reference)
    if ref is None:
        return None
    coarse_match = _first_match(match_contact_episodes(reference=(ref,), comparison=coarse, config=matching))
    adaptive_match = _first_match(match_contact_episodes(reference=(ref,), comparison=adaptive, config=matching))
    coarse_ep = _episode_by_match(coarse, coarse_match)
    adaptive_ep = _episode_by_match(adaptive, adaptive_match)
    coarse_re = _optional_abs(ref.restitution, None if coarse_ep is None else coarse_ep.restitution)
    adaptive_re = _optional_abs(ref.restitution, None if adaptive_ep is None else adaptive_ep.restitution)
    coarse_pe = None if coarse_ep is None else abs(ref.maximum_penetration - coarse_ep.maximum_penetration)
    adaptive_pe = None if adaptive_ep is None else abs(ref.maximum_penetration - adaptive_ep.maximum_penetration)
    return PrimaryImpactBenchmarkComparison(
        case_id=case_id,
        candidate_id=ref.candidate_id,
        reference_episode=ref,
        coarse_episode=coarse_ep,
        adaptive_episode=adaptive_ep,
        coarse_match_status=coarse_match.status if coarse_match else EpisodeMatchStatus.UNMATCHED_REFERENCE,
        adaptive_match_status=adaptive_match.status if adaptive_match else EpisodeMatchStatus.UNMATCHED_REFERENCE,
        coarse_restitution_error=coarse_re,
        adaptive_restitution_error=adaptive_re,
        coarse_penetration_error=coarse_pe,
        adaptive_penetration_error=adaptive_pe,
        coarse_duration_error=_optional_abs(ref.duration_seconds, None if coarse_ep is None else coarse_ep.duration_seconds),
        adaptive_duration_error=_optional_abs(ref.duration_seconds, None if adaptive_ep is None else adaptive_ep.duration_seconds),
        adaptive_improves_restitution=None if coarse_re is None or adaptive_re is None else adaptive_re <= coarse_re,
        adaptive_improves_penetration=None if coarse_pe is None or adaptive_pe is None else adaptive_pe <= coarse_pe,
    )


def run_episode_reference_convergence(
    case: ContactBenchmarkCase,
    *,
    segmentation: ContactEpisodeSegmentationConfig,
    recommendation: SubstepRecommendationConfig = SubstepRecommendationConfig(maximum_substeps=16),
    convergence_config: ReferenceConvergenceConfig = ReferenceConvergenceConfig(),
    matching_config: EpisodeMatchingConfig | None = None,
) -> EpisodeReferenceConvergenceResult:
    """Run episode-level fine/finer/ultra-fine convergence for primary impact."""
    matching = matching_config or EpisodeMatchingConfig(maximum_start_time_difference=case.macro_timestep)
    levels_episodes: list[tuple[int, float, tuple[ContactEpisodeMetrics, ...]]] = []
    for level_index, factor in enumerate(convergence_config.refinement_factors):
        level_recommendation = SubstepRecommendationConfig(maximum_substeps=recommendation.maximum_substeps * factor)
        samples = collect_contact_episode_samples(case, mode=BenchmarkMode.FIXED_FINE, recommendation=level_recommendation)
        episodes = segment_contact_episodes(samples, config=segmentation)
        levels_episodes.append((level_index, case.macro_timestep / level_recommendation.maximum_substeps, episodes))
    reference_primary = _primary(levels_episodes[-1][2])
    rows: list[EpisodeReferenceLevelResult] = []
    for level_index, timestep, episodes in levels_episodes:
        primary = _primary(episodes)
        status = EpisodeMatchStatus.UNMATCHED_REFERENCE
        if reference_primary is not None and primary is not None:
            match = _first_match(match_contact_episodes(reference=(reference_primary,), comparison=(primary,), config=matching))
            status = EpisodeMatchStatus.UNMATCHED_REFERENCE if match is None else match.status
        if primary is not None:
            rows.append(
                EpisodeReferenceLevelResult(
                    refinement_level=level_index,
                    timestep=timestep,
                    episode_index=primary.episode_index,
                    episode_kind=primary.kind,
                    start_time=primary.start_time,
                    restitution=primary.restitution,
                    maximum_penetration=primary.maximum_penetration,
                    contact_duration_seconds=primary.duration_seconds,
                    match_status=status,
                )
            )
    invalid = len(rows) < 3 or any(row.match_status is not EpisodeMatchStatus.MATCHED for row in rows[:-1])
    restitution = _metric_convergence("episode_restitution", tuple(row.restitution for row in rows), convergence_config.restitution_absolute_tolerance, convergence_config.restitution_relative_tolerance, invalid=False)
    penetration = _metric_convergence("episode_maximum_penetration", tuple(row.maximum_penetration for row in rows), convergence_config.penetration_absolute_tolerance, convergence_config.penetration_relative_tolerance, invalid=False)
    duration = _metric_convergence("episode_contact_duration", tuple(row.contact_duration_seconds for row in rows), convergence_config.duration_absolute_tolerance, convergence_config.duration_relative_tolerance, invalid=False, required=False)
    start_time = _metric_convergence("episode_start_time", tuple(row.start_time for row in rows), convergence_config.duration_absolute_tolerance, convergence_config.duration_relative_tolerance, invalid=False)
    if invalid:
        overall = ReferenceConvergenceStatus.INVALID_RESULT
    elif restitution.status is ReferenceConvergenceStatus.CONVERGED and penetration.status is ReferenceConvergenceStatus.CONVERGED:
        overall = ReferenceConvergenceStatus.CONVERGED
    else:
        overall = ReferenceConvergenceStatus.NOT_CONVERGED
    candidate_id = reference_primary.candidate_id if reference_primary is not None else _benchmark._candidate_for_case(case).candidate_id
    return EpisodeReferenceConvergenceResult(
        case_id=case.case_id,
        candidate_id=candidate_id,
        episode_kind=ContactEpisodeKind.PRIMARY_IMPACT,
        levels=tuple(rows),
        restitution=restitution,
        maximum_penetration=penetration,
        contact_duration=duration,
        start_time=start_time,
        overall_status=overall,
    )


def build_contact_episode_statistics(
    episodes: Sequence[ContactEpisodeMetrics],
    raw_intervals: Sequence[RawContactInterval],
    *,
    unmatched_episode_count: int = 0,
) -> ContactEpisodeStatistics:
    """Summarize contact intervals and merged episodes."""
    counts = Counter(episode.kind for episode in episodes)
    return ContactEpisodeStatistics(
        raw_interval_count=len(raw_intervals),
        merged_episode_count=len(episodes),
        primary_impact_count=counts[ContactEpisodeKind.PRIMARY_IMPACT],
        secondary_impact_count=counts[ContactEpisodeKind.SECONDARY_IMPACT],
        resting_contact_count=counts[ContactEpisodeKind.RESTING_CONTACT],
        chatter_interval_count=counts[ContactEpisodeKind.CONTACT_CHATTER],
        merged_chatter_gap_count=sum(episode.merged_gap_count for episode in episodes),
        unmatched_episode_count=unmatched_episode_count,
    )


def export_contact_episodes_csv(rows: Sequence[tuple[str, BenchmarkMode, ContactEpisodeMetrics]], path: str | Path) -> None:
    """Export one CSV row per contact episode."""
    _write_csv([
        {
            "case_id": case_id,
            "mode": mode.value,
            "candidate_id": ep.candidate_id,
            "episode_index": ep.episode_index,
            "kind": ep.kind.value,
            "start_time": ep.start_time,
            "end_time": ep.end_time,
            "duration_seconds": ep.duration_seconds,
            "impact_speed": ep.impact_speed,
            "separation_speed": ep.separation_speed,
            "restitution": ep.restitution,
            "maximum_penetration": ep.maximum_penetration,
            "minimum_timestep": ep.minimum_timestep,
            "maximum_substep_count": ep.maximum_substep_count,
            "raw_interval_count": ep.raw_interval_count,
            "merged_gap_count": ep.merged_gap_count,
            "validity": ep.validity.value,
        }
        for case_id, mode, ep in rows
    ], path)


def export_episode_matches_csv(matches: Sequence[EpisodeMatch], path: str | Path) -> None:
    """Export episode match rows."""
    _write_csv([_match_row(match) for match in matches], path)


def export_episode_comparisons_csv(comparisons: Sequence[EpisodeMetricComparison], path: str | Path) -> None:
    """Export matched episode comparison rows."""
    _write_csv([
        {
            **asdict(comp),
            "reference_mode": comp.reference_mode.value,
            "comparison_mode": comp.comparison_mode.value,
            "kind": comp.kind.value,
            "reference_validity": comp.reference_validity.value,
            "comparison_validity": comp.comparison_validity.value,
        }
        for comp in comparisons
    ], path)


def export_episode_diagnostics_json(dataset: ContactEpisodeAnalysisDataset, path: str | Path) -> None:
    """Export full episode diagnostics JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_dataset_to_dict(dataset), indent=2, sort_keys=True), encoding="utf8")


def write_episode_markdown_report(dataset: ContactEpisodeAnalysisDataset, path: str | Path) -> None:
    """Write a Markdown report for contact episode analysis."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_episode_markdown_report(dataset), encoding="utf8")


def build_episode_markdown_report(dataset: ContactEpisodeAnalysisDataset) -> str:
    """Build a Markdown report for episode diagnostics."""
    all_stats = dataset.statistics_by_mode
    lines = [
        "# Contact Episode Analysis",
        "",
        "## Overview",
        "",
    ]
    for mode, stats in sorted(all_stats.items()):
        lines.append(
            f"- {mode}: raw intervals={stats.raw_interval_count}, episodes={stats.merged_episode_count}, "
            f"primary={stats.primary_impact_count}, secondary={stats.secondary_impact_count}, "
            f"resting={stats.resting_contact_count}, chatter={stats.chatter_interval_count}"
        )
    lines.extend(["", "## Primary Impact Comparison", ""])
    primary = dataset.primary_comparison
    if primary is None:
        lines.append("No reference primary impact was found.")
    else:
        lines.append(
            f"- reference episode {primary.reference_episode.episode_index}: e={_fmt(primary.reference_episode.restitution)}, "
            f"penetration={_fmt(primary.reference_episode.maximum_penetration)}, duration={_fmt(primary.reference_episode.duration_seconds)}"
        )
        lines.append(f"- coarse match: {primary.coarse_match_status.value}, e error={_fmt(primary.coarse_restitution_error)}")
        lines.append(f"- adaptive match: {primary.adaptive_match_status.value}, e error={_fmt(primary.adaptive_restitution_error)}")
    failures = [match for match in dataset.matches if match.status is not EpisodeMatchStatus.MATCHED]
    lines.extend(["", "## Episode Matching Failures", ""])
    lines.extend([f"- {match.status.value}: {match.reason}" for match in failures] or ["- none"])
    lines.extend(["", "## Chatter Analysis", ""])
    for mode, stats in sorted(all_stats.items()):
        lines.append(f"- {mode}: merged chatter gaps={stats.merged_chatter_gap_count}")
    lines.extend(["", "## Reference Convergence", ""])
    if dataset.reference_convergence is None:
        lines.append("- not run")
    else:
        lines.append(f"- primary-impact convergence: {dataset.reference_convergence.overall_status.value}")
    lines.extend(["", "## Conclusion", "", _episode_conclusion(dataset)])
    return "\n".join(lines) + "\n"


def _collect_adaptive_samples(
    case: ContactBenchmarkCase,
    backend: MuJoCoBackend,
    recommendation: SubstepRecommendationConfig,
    macro_steps: int,
) -> tuple[ContactEpisodeSample, ...]:
    runner = AdaptiveMuJoCoRunner(
        backend,
        candidates=(_benchmark._candidate_for_case(case),),
        config=AdaptiveSubstepConfig(
            macro_timestep=case.macro_timestep,
            recommendation=recommendation,
            resting_window_macro_steps=3,
            separating_hold_macro_steps=1,
        ),
    )
    result = runner.reset()
    samples = [_sample_from_result(case, result, macro_step_index=0, substep_count=1, timestep=case.macro_timestep, adaptive_state=ContactMotionState.FREE)]
    for _ in range(macro_steps):
        adaptive = runner.step()
        for sample in adaptive.substep_results:
            samples.append(
                _sample_from_result(
                    case,
                    sample,
                    macro_step_index=adaptive.advance_result.macro_step_index,
                    substep_count=adaptive.decision.substep_count,
                    timestep=adaptive.decision.actual_substep_timestep,
                    adaptive_state=adaptive.decision.state_after,
                )
            )
    return tuple(samples)


def _sample_from_result(
    case: ContactBenchmarkCase,
    result: SimulationStepResult,
    *,
    macro_step_index: int | None,
    substep_count: int,
    timestep: float,
    adaptive_state: ContactMotionState | None,
) -> ContactEpisodeSample:
    candidate = _benchmark._candidate_for_case(case)
    if isinstance(case, SpherePlaneBenchmarkCase):
        sphere = result.get_body_state("sphere_01/sphere")
        body_a_id = "ground_01/ground"
        body_b_id = "sphere_01/sphere"
        normal = (0.0, 0.0, 1.0)
        body_a_linear = (0.0, 0.0, 0.0)
        body_a_angular = (0.0, 0.0, 0.0)
        body_b_linear = sphere.linear_velocity
        body_b_angular = sphere.angular_velocity
        gap = sphere.position[2] - case.radius
        normal_velocity = _dot(body_b_linear, normal)
    else:
        first = result.get_body_state("sphere_a_01/sphere_a")
        second = result.get_body_state("sphere_b_01/sphere_b")
        body_a_id = first.body_id
        body_b_id = second.body_id
        offset = _sub(second.position, first.position)
        normal = _normalize(offset)
        gap = _norm(offset) - 2.0 * case.radius
        body_a_linear = first.linear_velocity
        body_b_linear = second.linear_velocity
        body_a_angular = first.angular_velocity
        body_b_angular = second.angular_velocity
        normal_velocity = _dot(_sub(body_b_linear, body_a_linear), normal)
    contacts = _benchmark._contacts_for_case(case, result)
    penetration = max((contact.penetration_depth for contact in contacts), default=max(0.0, -gap))
    return ContactEpisodeSample(
        simulation_time=result.time,
        physics_step_index=result.step_index,
        macro_step_index=macro_step_index,
        candidate_id=candidate.candidate_id,
        body_a_id=body_a_id,
        body_b_id=body_b_id,
        active_contact=bool(contacts),
        gap_or_distance=gap,
        penetration=penetration,
        contact_normal=normal,
        normal_relative_velocity=normal_velocity,
        body_a_linear_velocity=body_a_linear,
        body_b_linear_velocity=body_b_linear,
        body_a_angular_velocity=body_a_angular,
        body_b_angular_velocity=body_b_angular,
        substep_count=substep_count,
        timestep=timestep,
        adaptive_state=adaptive_state,
    )


def _merge_reason(
    samples: Sequence[ContactEpisodeSample],
    previous: RawContactInterval,
    current: RawContactInterval,
    config: ContactEpisodeSegmentationConfig,
) -> IntervalMergeReason:
    gap_samples = tuple(samples[previous.end_sample_index + 1 : current.start_sample_index])
    gap_duration = max(0.0, current.start_time - previous.end_time)
    gap_steps = max(0, current.start_sample_index - previous.end_sample_index - 1)
    if gap_duration > config.maximum_chatter_gap_seconds or gap_steps > config.maximum_chatter_gap_steps:
        return IntervalMergeReason.NONE
    separated = any((sample.normal_relative_velocity or 0.0) >= config.minimum_separation_speed for sample in gap_samples)
    if not separated:
        return IntervalMergeReason.CONTACT_CHATTER
    if config.minimum_separation_duration <= 0.0:
        return IntervalMergeReason.NONE
    duration = sum(sample.timestep for sample in gap_samples if (sample.normal_relative_velocity or 0.0) >= config.minimum_separation_speed)
    return IntervalMergeReason.NONE if duration >= config.minimum_separation_duration else IntervalMergeReason.INSUFFICIENT_SEPARATION


def _build_episode(
    all_samples: Sequence[ContactEpisodeSample],
    intervals: Sequence[RawContactInterval],
    merged_reasons: Sequence[IntervalMergeReason],
    config: ContactEpisodeSegmentationConfig,
    max_restitution: float,
    max_norm_pen: float,
) -> ContactEpisodeMetrics:
    first = intervals[0]
    last = intervals[-1]
    episode_samples = tuple(all_samples[first.start_sample_index : last.end_sample_index + 1])
    active = tuple(sample for sample in episode_samples if sample.active_contact)
    pre = _last_before(all_samples, first.start_sample_index)
    post = _first_after(all_samples, last.end_sample_index)
    pre_v = None if pre is None else pre.normal_relative_velocity
    post_v = None if post is None else post.normal_relative_velocity
    first_contact_v = active[0].normal_relative_velocity if active else None
    impact_source = pre_v if pre_v is not None and pre_v < 0.0 else first_contact_v
    impact = None if impact_source is None or impact_source >= 0.0 else -impact_source
    separation = None if post_v is None or post_v <= 0.0 or last.ended_while_contact_active else post_v
    restitution = None if impact is None or impact <= 0.0 or separation is None else separation / impact
    max_sample = max(active, key=lambda sample: (sample.penetration, -sample.simulation_time))
    norm_pen = max_sample.penetration / _char_length(first, active)
    validity = classify_benchmark_validity(
        outcome=RestitutionOutcome.REBOUNDED if restitution is not None else RestitutionOutcome.TIMEOUT,
        restitution=restitution,
        normalized_penetration=norm_pen,
        validation=_Validation(max_restitution, max_norm_pen),
    )
    return ContactEpisodeMetrics(
        episode_index=0,
        candidate_id=first.candidate_id,
        kind=ContactEpisodeKind.UNCLASSIFIED,
        body_a_id=first.body_a_id,
        body_b_id=first.body_b_id,
        start_time=first.start_time,
        end_time=last.end_time,
        duration_seconds=max(0.0, last.end_time - first.start_time),
        raw_interval_count=len(intervals),
        merged_gap_count=len(merged_reasons),
        pre_contact_normal_velocity=pre_v,
        impact_speed=impact,
        post_contact_normal_velocity=post_v,
        separation_speed=separation,
        restitution=restitution,
        maximum_penetration=max_sample.penetration,
        normalized_penetration=norm_pen,
        maximum_penetration_time=max_sample.simulation_time,
        contact_sample_count=len(active),
        physics_step_count=max(0, last.end_sample_index - first.start_sample_index + 1),
        minimum_timestep=min(sample.timestep for sample in episode_samples),
        maximum_timestep=max(sample.timestep for sample in episode_samples),
        minimum_substep_count=min(sample.substep_count for sample in episode_samples),
        maximum_substep_count=max(sample.substep_count for sample in episode_samples),
        state_at_start=first.samples[0].adaptive_state,
        state_at_maximum_penetration=max_sample.adaptive_state,
        state_at_end=last.samples[-1].adaptive_state,
        ended_with_separation=separation is not None,
        ended_while_contact_active=last.ended_while_contact_active,
        validity=validity,
    )


@dataclass(frozen=True)
class _Validation:
    maximum_restitution: float
    maximum_normalized_penetration: float


def _classify_episodes(
    episodes: Sequence[ContactEpisodeMetrics],
    config: ContactEpisodeSegmentationConfig,
) -> list[ContactEpisodeMetrics]:
    classified: list[ContactEpisodeMetrics] = []
    primary_seen: set[str] = set()
    for index, episode in enumerate(sorted(episodes, key=lambda item: (item.candidate_id, item.start_time))):
        kind = ContactEpisodeKind.UNCLASSIFIED
        if episode.duration_seconds >= config.resting_duration_threshold and _low_speed_episode(episode, config):
            kind = ContactEpisodeKind.RESTING_CONTACT
        elif episode.impact_speed is not None and episode.impact_speed >= config.minimum_impact_speed:
            if episode.candidate_id not in primary_seen:
                kind = ContactEpisodeKind.PRIMARY_IMPACT
                primary_seen.add(episode.candidate_id)
            else:
                kind = ContactEpisodeKind.SECONDARY_IMPACT
        elif episode.duration_seconds <= config.minimum_episode_duration or episode.raw_interval_count > 1:
            kind = ContactEpisodeKind.CONTACT_CHATTER
        classified.append(_replace_episode(episode, episode_index=index, kind=kind))
    return classified


def _replace_episode(episode: ContactEpisodeMetrics, **updates) -> ContactEpisodeMetrics:
    data = episode.__dict__.copy()
    data.update(updates)
    return ContactEpisodeMetrics(**data)


def _low_speed_episode(episode: ContactEpisodeMetrics, config: ContactEpisodeSegmentationConfig) -> bool:
    speeds = [abs(value) for value in (episode.pre_contact_normal_velocity, episode.post_contact_normal_velocity) if value is not None]
    return bool(speeds) and max(speeds) <= config.resting_speed_threshold and not episode.ended_with_separation


def _last_before(samples: Sequence[ContactEpisodeSample], index: int) -> ContactEpisodeSample | None:
    for sample in reversed(samples[:index]):
        if sample.normal_relative_velocity is not None:
            return sample
    return None


def _first_after(samples: Sequence[ContactEpisodeSample], index: int) -> ContactEpisodeSample | None:
    for sample in samples[index + 1 :]:
        if sample.normal_relative_velocity is not None:
            return sample
    return None


def _char_length(interval: RawContactInterval, samples: Sequence[ContactEpisodeSample]) -> float:
    if "sphere_a" in interval.body_a_id and "sphere_b" in interval.body_b_id:
        return 0.1
    return 0.1


def _unmatched_reference(ref: ContactEpisodeMetrics, reason: str) -> EpisodeMatch:
    return EpisodeMatch(ref.candidate_id, ref.episode_index, None, EpisodeMatchStatus.UNMATCHED_REFERENCE, ref.kind, None, None, None, reason)


def _compatible_kind(first: ContactEpisodeKind, second: ContactEpisodeKind) -> bool:
    if first is second:
        return True
    impact = {ContactEpisodeKind.PRIMARY_IMPACT, ContactEpisodeKind.SECONDARY_IMPACT}
    return first in impact and second in impact


def _kind_rank(kind: ContactEpisodeKind) -> int:
    order = {
        ContactEpisodeKind.PRIMARY_IMPACT: 0,
        ContactEpisodeKind.SECONDARY_IMPACT: 1,
        ContactEpisodeKind.RESTING_CONTACT: 2,
        ContactEpisodeKind.CONTACT_CHATTER: 3,
        ContactEpisodeKind.UNCLASSIFIED: 4,
    }
    return order[kind]


def _primary(episodes: Sequence[ContactEpisodeMetrics]) -> ContactEpisodeMetrics | None:
    for episode in sorted(episodes, key=lambda item: (item.start_time, item.episode_index)):
        if episode.kind is ContactEpisodeKind.PRIMARY_IMPACT:
            return episode
    return None


def _first_match(matches: Sequence[EpisodeMatch]) -> EpisodeMatch | None:
    return matches[0] if matches else None


def _episode_by_match(episodes: Sequence[ContactEpisodeMetrics], match: EpisodeMatch | None) -> ContactEpisodeMetrics | None:
    if match is None or match.comparison_episode_index is None or match.status is not EpisodeMatchStatus.MATCHED:
        return None
    return next((episode for episode in episodes if episode.episode_index == match.comparison_episode_index), None)


def _optional_abs(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return abs(first - second)


def _sub(first: Vector3, second: Vector3) -> Vector3:
    return tuple(first[index] - second[index] for index in range(3))  # type: ignore[return-value]


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(first[index] * second[index] for index in range(3))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector3) -> Vector3:
    length = _norm(vector)
    if length <= 1.0e-12:
        return (1.0, 0.0, 0.0)
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def _write_csv(rows: Sequence[dict[str, object]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _match_row(match: EpisodeMatch) -> dict[str, object]:
    return {
        "candidate_id": match.candidate_id,
        "reference_episode_index": match.reference_episode_index,
        "comparison_episode_index": match.comparison_episode_index,
        "status": match.status.value,
        "reference_kind": None if match.reference_kind is None else match.reference_kind.value,
        "comparison_kind": None if match.comparison_kind is None else match.comparison_kind.value,
        "start_time_difference": match.start_time_difference,
        "impact_speed_difference": match.impact_speed_difference,
        "reason": match.reason,
    }


def _dataset_to_dict(dataset: ContactEpisodeAnalysisDataset) -> dict[str, object]:
    return {
        "case_id": dataset.case_id,
        "episodes_by_mode": {
            mode: [_episode_to_dict(episode) for episode in episodes]
            for mode, episodes in dataset.episodes_by_mode.items()
        },
        "matches": [_match_row(match) for match in dataset.matches],
        "comparisons": [_comparison_to_dict(comp) for comp in dataset.comparisons],
        "primary_comparison": None if dataset.primary_comparison is None else _primary_to_dict(dataset.primary_comparison),
        "reference_convergence": None if dataset.reference_convergence is None else _episode_convergence_to_dict(dataset.reference_convergence),
        "statistics_by_mode": {mode: asdict(stats) for mode, stats in dataset.statistics_by_mode.items()},
        "field_semantics": {
            "normal_relative_velocity": "negative means approaching along the configured contact normal; positive means separating",
            "sphere_plane_normal": "plane to sphere",
            "sphere_sphere_normal": "body_a to body_b",
        },
    }


def _episode_to_dict(episode: ContactEpisodeMetrics) -> dict[str, object]:
    data = asdict(episode)
    data["kind"] = episode.kind.value
    data["validity"] = episode.validity.value
    for key in ("state_at_start", "state_at_maximum_penetration", "state_at_end"):
        if data[key] is not None:
            data[key] = data[key].value
    return data


def _comparison_to_dict(comp: EpisodeMetricComparison) -> dict[str, object]:
    data = asdict(comp)
    data["reference_mode"] = comp.reference_mode.value
    data["comparison_mode"] = comp.comparison_mode.value
    data["kind"] = comp.kind.value
    data["reference_validity"] = comp.reference_validity.value
    data["comparison_validity"] = comp.comparison_validity.value
    return data


def _primary_to_dict(primary: PrimaryImpactBenchmarkComparison) -> dict[str, object]:
    data = asdict(primary)
    data["reference_episode"] = _episode_to_dict(primary.reference_episode)
    data["coarse_episode"] = None if primary.coarse_episode is None else _episode_to_dict(primary.coarse_episode)
    data["adaptive_episode"] = None if primary.adaptive_episode is None else _episode_to_dict(primary.adaptive_episode)
    data["coarse_match_status"] = primary.coarse_match_status.value
    data["adaptive_match_status"] = primary.adaptive_match_status.value
    return data


def _episode_convergence_to_dict(result: EpisodeReferenceConvergenceResult) -> dict[str, object]:
    data = asdict(result)
    data["episode_kind"] = result.episode_kind.value
    data["overall_status"] = result.overall_status.value
    for metric_name in ("restitution", "maximum_penetration", "contact_duration", "start_time"):
        data[metric_name]["status"] = getattr(result, metric_name).status.value
    for level in data["levels"]:
        level["episode_kind"] = ContactEpisodeKind(level["episode_kind"]).value if isinstance(level["episode_kind"], str) else level["episode_kind"].value
        level["match_status"] = EpisodeMatchStatus(level["match_status"]).value if isinstance(level["match_status"], str) else level["match_status"].value
    return data


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.6g}"


def _episode_conclusion(dataset: ContactEpisodeAnalysisDataset) -> str:
    primary = dataset.primary_comparison
    if primary is None:
        return "No primary impact was available for comparison."
    if dataset.reference_convergence is not None and dataset.reference_convergence.overall_status is ReferenceConvergenceStatus.CONVERGED:
        return "Primary-impact reference is converged for the selected refinement levels."
    return "Episode-level diagnostics separate primary-impact behavior from later contact episodes."
