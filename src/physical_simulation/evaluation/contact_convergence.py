"""Reference convergence and failure attribution for adaptive contact benchmarks."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation.contact_calibration import RestitutionOutcome
from physical_simulation.evaluation.contact_benchmark import (
    BenchmarkComparison,
    BenchmarkMode,
    BenchmarkValidationConfig,
    BenchmarkValidity,
    ContactBenchmarkCase,
    ContactBenchmarkDataset,
    ContactBenchmarkResult,
    SpherePlaneBenchmarkCase,
    classify_benchmark_validity,
)
import physical_simulation.evaluation.contact_benchmark as _benchmark
from physical_simulation.mujoco import (
    AdaptiveMuJoCoRunner,
    AdaptiveSubstepConfig,
    ContactMotionState,
    SubstepRecommendationConfig,
)
from physical_simulation.validation.asset_validator import _finite_float
from physical_simulation.validation.errors import PhysicsValidationError


class ReferenceConvergenceStatus(Enum):
    """Status for fixed-fine reference refinement checks."""

    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    INSUFFICIENT_LEVELS = "insufficient_levels"
    INVALID_RESULT = "invalid_result"


@dataclass(frozen=True)
class ReferenceLevelResult:
    """One fixed-timestep refinement level result."""

    refinement_level: int
    timestep: float
    refinement_factor: int
    restitution: float | None
    rebound_speed: float | None
    maximum_penetration: float
    contact_duration_seconds: float | None
    validity: BenchmarkValidity
    physics_step_count: int


@dataclass(frozen=True)
class ReferenceMetricConvergence:
    """Convergence diagnostic for one scalar metric."""

    metric_name: str
    coarse_to_fine_difference: float | None
    fine_to_finer_difference: float | None
    difference_ratio: float | None
    absolute_tolerance: float
    relative_tolerance: float
    status: ReferenceConvergenceStatus


@dataclass(frozen=True)
class ReferenceConvergenceResult:
    """Reference convergence result for one benchmark case."""

    case_id: str
    levels: tuple[ReferenceLevelResult, ...]
    restitution: ReferenceMetricConvergence
    rebound_speed: ReferenceMetricConvergence
    maximum_penetration: ReferenceMetricConvergence
    contact_duration: ReferenceMetricConvergence
    overall_status: ReferenceConvergenceStatus
    selected_reference_level: int | None


@dataclass(frozen=True)
class ReferenceConvergenceConfig:
    """Configuration for reference timestep refinement."""

    refinement_factors: tuple[int, ...] = (1, 2, 4)
    restitution_absolute_tolerance: float = 0.005
    restitution_relative_tolerance: float = 0.02
    rebound_speed_absolute_tolerance: float = 0.01
    rebound_speed_relative_tolerance: float = 0.02
    penetration_absolute_tolerance: float = 5.0e-4
    penetration_relative_tolerance: float = 0.05
    duration_absolute_tolerance: float = 1.0e-3
    duration_relative_tolerance: float = 0.05

    def __post_init__(self) -> None:
        factors = tuple(self.refinement_factors)
        if len(factors) < 1 or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in factors):
            raise PhysicsValidationError(f"refinement_factors must contain positive ints; actual value={factors!r}")
        if len(set(factors)) != len(factors):
            raise PhysicsValidationError(f"refinement_factors must be unique; actual value={factors!r}")
        object.__setattr__(self, "refinement_factors", tuple(sorted(factors)))
        for field_name in (
            "restitution_absolute_tolerance",
            "restitution_relative_tolerance",
            "rebound_speed_absolute_tolerance",
            "rebound_speed_relative_tolerance",
            "penetration_absolute_tolerance",
            "penetration_relative_tolerance",
            "duration_absolute_tolerance",
            "duration_relative_tolerance",
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
class AdaptiveContactEpisodeTrace:
    """Trace for one predicted or observed contact episode."""

    episode_index: int
    candidate_id: str
    prediction_time: float | None
    predicted_absolute_contact_time: float | None
    first_actual_contact_time: float | None
    prediction_lead_time_seconds: float | None
    prediction_lead_time_macro_steps: float | None
    approaching_start_time: float | None
    impacting_start_time: float | None
    separating_start_time: float | None
    resting_start_time: float | None
    episode_end_time: float | None
    gap_at_first_prediction: float | None
    approach_speed_at_first_prediction: float | None
    solver_characteristic_timescale: float | None
    requested_substep_count: int | None
    maximum_actual_substep_count: int
    limited_by_maximum_substeps: bool
    minimum_actual_timestep: float
    substepped_macro_step_count: int
    contact_substep_count: int
    maximum_penetration: float
    maximum_penetration_time: float | None
    state_at_maximum_penetration: ContactMotionState | None
    contact_duration_seconds: float | None


@dataclass(frozen=True)
class AdaptiveDiagnosticTrace:
    """Run-level trace for one adaptive benchmark case."""

    case_id: str
    episodes: tuple[AdaptiveContactEpisodeTrace, ...]
    total_contact_episode_count: int
    total_substepped_macro_steps: int
    total_physics_steps: int
    first_prediction_time: float | None
    first_contact_time: float | None
    global_minimum_timestep: float
    global_maximum_substep_count: int


class AdaptiveFailureReason(Enum):
    """Deterministic reasons used to explain adaptive non-improvement."""

    NONE = "none"
    LATE_PREDICTION = "late_prediction"
    SHORT_PREDICTION_LEAD = "short_prediction_lead"
    MAX_SUBSTEPS_LIMITED = "max_substeps_limited"
    INSUFFICIENT_TIME_RESOLUTION = "insufficient_time_resolution"
    EARLY_FINE_EXIT = "early_fine_exit"
    MULTIPLE_CONTACT_EPISODES = "multiple_contact_episodes"
    REFERENCE_NOT_CONVERGED = "reference_not_converged"
    METRIC_SAMPLING_SENSITIVITY = "metric_sampling_sensitivity"
    NONPHYSICAL_ADAPTIVE_RESULT = "nonphysical_adaptive_result"
    EPISODE_MISMATCH = "episode_mismatch"
    CONTACT_CHATTER = "contact_chatter"
    PRIMARY_IMPACT_NOT_FOUND = "primary_impact_not_found"
    PRIMARY_RESTITUTION_NOT_IMPROVED = "primary_restitution_not_improved"
    PRIMARY_PENETRATION_NOT_IMPROVED = "primary_penetration_not_improved"
    PRIMARY_DURATION_NOT_IMPROVED = "primary_duration_not_improved"
    RUN_LEVEL_SECONDARY_EPISODE_DIFFERENCE = "run_level_secondary_episode_difference"
    UNKNOWN = "unknown"


class ImprovementOutcome(Enum):
    """Outcome for adaptive-vs-coarse comparison against a reference."""

    IMPROVED = "improved"
    NOT_IMPROVED = "not_improved"
    BOTH_ACCEPTABLE = "both_acceptable"
    REFERENCE_UNRESOLVED = "reference_unresolved"
    EPISODE_UNMATCHED = "episode_unmatched"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class AdaptiveImprovementConfig:
    """Tolerances for deciding whether both methods are already acceptable."""

    restitution_error_tolerance: float = 0.005
    penetration_error_tolerance: float = 5.0e-4


@dataclass(frozen=True)
class AdaptiveFailureAttribution:
    """Failure attribution for one adaptive benchmark case."""

    case_id: str
    restitution_improved: bool | None
    penetration_improved: bool | None
    primary_reason: AdaptiveFailureReason
    secondary_reasons: tuple[AdaptiveFailureReason, ...]
    evidence: tuple[str, ...]
    restitution_outcome: ImprovementOutcome = ImprovementOutcome.NOT_APPLICABLE
    penetration_outcome: ImprovementOutcome = ImprovementOutcome.NOT_APPLICABLE


@dataclass(frozen=True)
class ConvergenceSelectionConfig:
    """Select which cases receive expensive reference convergence checks."""

    top_k_restitution_error: int = 5
    top_k_penetration_error: int = 5
    include_all_not_improved: bool = True
    include_nonphysical_coarse_cases: bool = True


@dataclass(frozen=True)
class AttributionBenchmarkSummary:
    """Summary of convergence and attribution diagnostics."""

    total_cases: int
    convergence_checked_cases: int
    converged_reference_cases: int
    unresolved_reference_cases: int
    failure_reason_counts: Mapping[str, int]
    improvement_outcome_counts: Mapping[str, int]
    mean_prediction_lead_macro_steps: float | None
    minimum_prediction_lead_macro_steps: float | None
    max_substep_limited_case_count: int
    multiple_episode_case_count: int
    early_fine_exit_case_count: int
    latest_prediction_case_id: str | None = None
    lowest_prediction_lead_case_id: str | None = None
    maximum_adaptive_restitution_error_case_id: str | None = None
    maximum_adaptive_penetration_error_case_id: str | None = None
    least_converged_reference_case_id: str | None = None


@dataclass(frozen=True)
class AttributionDiagnosticDataset:
    """Complete Phase 2G5 diagnostic dataset."""

    benchmark: ContactBenchmarkDataset
    convergence: tuple[ReferenceConvergenceResult, ...]
    traces: tuple[AdaptiveDiagnosticTrace, ...]
    attributions: tuple[AdaptiveFailureAttribution, ...]
    summary: AttributionBenchmarkSummary
    config: dict[str, object]
    units: dict[str, str]


def run_reference_convergence(
    case: ContactBenchmarkCase,
    *,
    recommendation: SubstepRecommendationConfig = SubstepRecommendationConfig(maximum_substeps=16),
    validation: BenchmarkValidationConfig = BenchmarkValidationConfig(),
    config: ReferenceConvergenceConfig = ReferenceConvergenceConfig(),
) -> ReferenceConvergenceResult:
    """Run fixed fine/finer/ultra-fine reference levels for one case."""
    base_substeps = recommendation.maximum_substeps
    macro_steps = _benchmark._macro_steps(case)
    base_timestep = case.macro_timestep / base_substeps
    levels: list[ReferenceLevelResult] = []
    for level_index, factor in enumerate(config.refinement_factors):
        timestep = base_timestep / factor
        steps = macro_steps * base_substeps * factor
        backend = MuJoCoBackend()
        try:
            backend.load_scene(_benchmark._scene_for_case(case, timestep=timestep))
            _benchmark._apply_initial_velocity(case, backend, update_initial=True)
            samples = _benchmark._run_fixed(case, backend, steps)
            measurement = _benchmark._measure_samples(case, samples)
            validity = classify_benchmark_validity(
                outcome=measurement.outcome,
                restitution=measurement.measured_restitution,
                normalized_penetration=measurement.normalized_penetration,
                validation=validation,
            )
            levels.append(
                ReferenceLevelResult(
                    refinement_level=level_index,
                    timestep=timestep,
                    refinement_factor=factor,
                    restitution=measurement.measured_restitution,
                    rebound_speed=measurement.rebound_speed,
                    maximum_penetration=measurement.maximum_penetration_depth,
                    contact_duration_seconds=measurement.contact_duration_seconds,
                    validity=validity,
                    physics_step_count=steps,
                )
            )
        except Exception:
            levels.append(
                ReferenceLevelResult(
                    refinement_level=level_index,
                    timestep=timestep,
                    refinement_factor=factor,
                    restitution=None,
                    rebound_speed=None,
                    maximum_penetration=math.nan,
                    contact_duration_seconds=None,
                    validity=BenchmarkValidity.UNSTABLE,
                    physics_step_count=0,
                )
            )
        finally:
            backend.close()
    return build_reference_convergence_result(case.case_id, tuple(levels), config=config)


def build_reference_convergence_result(
    case_id: str,
    levels: Sequence[ReferenceLevelResult],
    *,
    config: ReferenceConvergenceConfig = ReferenceConvergenceConfig(),
) -> ReferenceConvergenceResult:
    """Build convergence diagnostics from precomputed levels."""
    level_tuple = tuple(levels)
    invalid = any(level.validity is BenchmarkValidity.UNSTABLE or _bad(level.maximum_penetration) for level in level_tuple)
    restitution = _metric_convergence(
        "restitution",
        tuple(level.restitution for level in level_tuple),
        config.restitution_absolute_tolerance,
        config.restitution_relative_tolerance,
        invalid=invalid,
    )
    rebound = _metric_convergence(
        "rebound_speed",
        tuple(level.rebound_speed for level in level_tuple),
        config.rebound_speed_absolute_tolerance,
        config.rebound_speed_relative_tolerance,
        invalid=invalid,
    )
    penetration = _metric_convergence(
        "maximum_penetration",
        tuple(level.maximum_penetration for level in level_tuple),
        config.penetration_absolute_tolerance,
        config.penetration_relative_tolerance,
        invalid=invalid,
    )
    duration = _metric_convergence(
        "contact_duration",
        tuple(level.contact_duration_seconds for level in level_tuple),
        config.duration_absolute_tolerance,
        config.duration_relative_tolerance,
        invalid=invalid,
        required=False,
    )
    if len(level_tuple) < 3:
        overall = ReferenceConvergenceStatus.INSUFFICIENT_LEVELS
    elif invalid:
        overall = ReferenceConvergenceStatus.INVALID_RESULT
    elif (
        restitution.status is ReferenceConvergenceStatus.CONVERGED
        and penetration.status is ReferenceConvergenceStatus.CONVERGED
    ):
        overall = ReferenceConvergenceStatus.CONVERGED
    else:
        overall = ReferenceConvergenceStatus.NOT_CONVERGED
    selected = level_tuple[-1].refinement_level if overall is ReferenceConvergenceStatus.CONVERGED and level_tuple else None
    return ReferenceConvergenceResult(
        case_id=case_id,
        levels=level_tuple,
        restitution=restitution,
        rebound_speed=rebound,
        maximum_penetration=penetration,
        contact_duration=duration,
        overall_status=overall,
        selected_reference_level=selected,
    )


def run_adaptive_diagnostic_trace(
    case: ContactBenchmarkCase,
    *,
    recommendation: SubstepRecommendationConfig = SubstepRecommendationConfig(maximum_substeps=16),
    adaptive_config: AdaptiveSubstepConfig | None = None,
) -> AdaptiveDiagnosticTrace:
    """Run adaptive mode and collect structured contact episode traces."""
    macro_steps = _benchmark._macro_steps(case)
    config = adaptive_config or AdaptiveSubstepConfig(
        macro_timestep=case.macro_timestep,
        recommendation=recommendation,
        resting_window_macro_steps=3,
        separating_hold_macro_steps=1,
    )
    backend = MuJoCoBackend()
    try:
        backend.load_scene(_benchmark._scene_for_case(case, timestep=case.macro_timestep))
        _benchmark._apply_initial_velocity(case, backend, update_initial=True)
        runner = AdaptiveMuJoCoRunner(
            backend,
            candidates=(_benchmark._candidate_for_case(case),),
            config=config,
        )
        samples = [runner.reset()]
        builder: _EpisodeBuilder | None = None
        builders: list[_EpisodeBuilder] = []
        substepped_macros = 0
        max_substeps = 1
        min_timestep = config.macro_timestep
        first_prediction_time: float | None = None
        first_contact_time: float | None = None
        for _ in range(macro_steps):
            result = runner.step()
            decision = result.decision
            macro_end = result.advance_result.simulation_result.time
            macro_start = macro_end - decision.macro_timestep
            has_contact = any(_benchmark._contacts_for_case(case, sample) for sample in result.substep_results)
            if has_contact and first_contact_time is None:
                first_contact_time = next(
                    sample.time for sample in result.substep_results if _benchmark._contacts_for_case(case, sample)
                )
            starts_episode = decision.prediction is not None or has_contact or decision.active_contact_observed
            if builder is None and starts_episode:
                builder = _EpisodeBuilder(
                    episode_index=len(builders),
                    candidate_id=decision.selected_candidate_id or _benchmark._candidate_for_case(case).candidate_id,
                    macro_timestep=config.macro_timestep,
                )
                builders.append(builder)
            if builder is not None:
                builder.observe_decision(decision, macro_start, macro_end)
                builder.observe_samples(case, result.substep_results, decision.state_after)
                if first_prediction_time is None and builder.prediction_time is not None:
                    first_prediction_time = builder.prediction_time
                if decision.substep_count > 1:
                    substepped_macros += 1
                max_substeps = max(max_substeps, decision.substep_count)
                min_timestep = min(min_timestep, decision.actual_substep_timestep)
                if decision.state_after in {ContactMotionState.FREE, ContactMotionState.RESTING} and not has_contact:
                    builder.episode_end_time = macro_end
                    builder = None
            samples.extend(result.substep_results or (result.advance_result.simulation_result,))
        if builder is not None:
            builder.episode_end_time = samples[-1].time
        episodes = tuple(item.build() for item in builders)
        return AdaptiveDiagnosticTrace(
            case_id=case.case_id,
            episodes=episodes,
            total_contact_episode_count=len(episodes),
            total_substepped_macro_steps=substepped_macros,
            total_physics_steps=runner.physics_step_count,
            first_prediction_time=first_prediction_time,
            first_contact_time=first_contact_time,
            global_minimum_timestep=min_timestep,
            global_maximum_substep_count=max_substeps,
        )
    finally:
        backend.close()


def select_convergence_cases(
    cases: Sequence[ContactBenchmarkCase],
    dataset: ContactBenchmarkDataset,
    *,
    config: ConvergenceSelectionConfig = ConvergenceSelectionConfig(),
) -> tuple[ContactBenchmarkCase, ...]:
    """Select a deterministic subset of cases for convergence refinement."""
    by_id = {case.case_id: case for case in cases}
    selected: set[str] = set()
    comparisons = tuple(sorted(dataset.comparisons, key=lambda item: item.case_id))
    if config.include_all_not_improved:
        selected.update(
            item.case_id for item in comparisons
            if item.adaptive_improves_restitution is False or not item.adaptive_improves_penetration
        )
    if config.include_nonphysical_coarse_cases:
        selected.update(
            result.case_id for result in dataset.results
            if result.mode is BenchmarkMode.FIXED_COARSE and result.validity is BenchmarkValidity.NONPHYSICAL_REBOUND
        )
    selected.update(_top_k(comparisons, config.top_k_restitution_error, lambda item: item.adaptive_restitution_error))
    selected.update(_top_k(comparisons, config.top_k_penetration_error, lambda item: item.adaptive_penetration_error))
    return tuple(by_id[case_id] for case_id in sorted(selected) if case_id in by_id)


def attribute_adaptive_failure(
    *,
    comparison: BenchmarkComparison,
    adaptive_result: ContactBenchmarkResult,
    trace: AdaptiveDiagnosticTrace,
    convergence: ReferenceConvergenceResult | None,
    recommendation: SubstepRecommendationConfig = SubstepRecommendationConfig(maximum_substeps=16),
    improvement_config: AdaptiveImprovementConfig = AdaptiveImprovementConfig(),
    short_prediction_lead_macro_steps: float = 0.5,
) -> AdaptiveFailureAttribution:
    """Assign deterministic failure-attribution reasons from structured diagnostics."""
    restitution_outcome = _improvement_outcome(
        comparison.coarse_restitution_error,
        comparison.adaptive_restitution_error,
        improvement_config.restitution_error_tolerance,
        convergence,
    )
    penetration_outcome = _improvement_outcome(
        comparison.coarse_penetration_error,
        comparison.adaptive_penetration_error,
        improvement_config.penetration_error_tolerance,
        convergence,
    )
    reasons: list[AdaptiveFailureReason] = []
    evidence: list[str] = []
    if adaptive_result.validity in {
        BenchmarkValidity.UNSTABLE,
        BenchmarkValidity.NONPHYSICAL_REBOUND,
    } or _bad(adaptive_result.restitution) or _bad(adaptive_result.maximum_penetration):
        reasons.append(AdaptiveFailureReason.NONPHYSICAL_ADAPTIVE_RESULT)
        evidence.append(f"adaptive validity = {adaptive_result.validity.value}")
    if convergence is not None and convergence.overall_status is not ReferenceConvergenceStatus.CONVERGED:
        reasons.append(AdaptiveFailureReason.REFERENCE_NOT_CONVERGED)
        evidence.append(f"reference status = {convergence.overall_status.value}")
    lead = _first_lead(trace)
    if lead is not None:
        if lead <= 0.0:
            reasons.append(AdaptiveFailureReason.LATE_PREDICTION)
            evidence.append(f"prediction lead = {lead:.3g} macro steps")
        elif lead < short_prediction_lead_macro_steps:
            reasons.append(AdaptiveFailureReason.SHORT_PREDICTION_LEAD)
            evidence.append(f"prediction lead = {lead:.3g} macro steps")
    elif trace.first_contact_time is not None:
        reasons.append(AdaptiveFailureReason.LATE_PREDICTION)
        evidence.append("contact was observed before any recorded prediction lead")
    if any(episode.limited_by_maximum_substeps for episode in trace.episodes):
        reasons.append(AdaptiveFailureReason.MAX_SUBSTEPS_LIMITED)
        evidence.append("maximum_substeps was reached")
    if _insufficient_resolution(trace, recommendation):
        reasons.append(AdaptiveFailureReason.INSUFFICIENT_TIME_RESOLUTION)
        evidence.append("actual samples per solver characteristic time were below target")
    if _early_fine_exit(trace):
        reasons.append(AdaptiveFailureReason.EARLY_FINE_EXIT)
        evidence.append("maximum penetration occurred after fine mode ended")
    if trace.total_contact_episode_count > 1:
        reasons.append(AdaptiveFailureReason.MULTIPLE_CONTACT_EPISODES)
        evidence.append(f"{trace.total_contact_episode_count} separate contact episodes were observed")
    if _metric_sampling_sensitive(convergence):
        reasons.append(AdaptiveFailureReason.METRIC_SAMPLING_SENSITIVITY)
        evidence.append("reference metrics show sampling-sensitive non-monotonic differences")
    reasons = _dedupe_reasons(reasons)
    primary = _primary_reason(reasons)
    if primary is AdaptiveFailureReason.NONE and (
        restitution_outcome is ImprovementOutcome.NOT_IMPROVED or penetration_outcome is ImprovementOutcome.NOT_IMPROVED
    ):
        primary = AdaptiveFailureReason.UNKNOWN
        evidence.append("adaptive did not improve at least one metric but no stronger diagnostic matched")
    return AdaptiveFailureAttribution(
        case_id=comparison.case_id,
        restitution_improved=comparison.adaptive_improves_restitution,
        penetration_improved=comparison.adaptive_improves_penetration,
        primary_reason=primary,
        secondary_reasons=tuple(reason for reason in reasons if reason is not primary),
        evidence=tuple(evidence),
        restitution_outcome=restitution_outcome,
        penetration_outcome=penetration_outcome,
    )


def build_attribution_summary(
    *,
    benchmark: ContactBenchmarkDataset,
    convergence: Sequence[ReferenceConvergenceResult],
    traces: Sequence[AdaptiveDiagnosticTrace],
    attributions: Sequence[AdaptiveFailureAttribution],
) -> AttributionBenchmarkSummary:
    """Build deterministic aggregate summary for attribution diagnostics."""
    convergence_tuple = tuple(convergence)
    trace_tuple = tuple(traces)
    attribution_tuple = tuple(attributions)
    leads = [lead for trace in trace_tuple for lead in (_first_lead(trace),) if lead is not None]
    reason_counts = Counter(item.primary_reason.value for item in attribution_tuple)
    for item in attribution_tuple:
        reason_counts.update(reason.value for reason in item.secondary_reasons)
    outcome_counts = Counter()
    for item in attribution_tuple:
        outcome_counts[item.restitution_outcome.value] += 1
        outcome_counts[item.penetration_outcome.value] += 1
    return AttributionBenchmarkSummary(
        total_cases=len(benchmark.cases),
        convergence_checked_cases=len(convergence_tuple),
        converged_reference_cases=sum(item.overall_status is ReferenceConvergenceStatus.CONVERGED for item in convergence_tuple),
        unresolved_reference_cases=sum(item.overall_status is not ReferenceConvergenceStatus.CONVERGED for item in convergence_tuple),
        failure_reason_counts=dict(sorted(reason_counts.items())),
        improvement_outcome_counts=dict(sorted(outcome_counts.items())),
        mean_prediction_lead_macro_steps=None if not leads else sum(leads) / len(leads),
        minimum_prediction_lead_macro_steps=None if not leads else min(leads),
        max_substep_limited_case_count=sum(any(ep.limited_by_maximum_substeps for ep in trace.episodes) for trace in trace_tuple),
        multiple_episode_case_count=sum(trace.total_contact_episode_count > 1 for trace in trace_tuple),
        early_fine_exit_case_count=sum(_early_fine_exit(trace) for trace in trace_tuple),
        latest_prediction_case_id=_min_case(trace_tuple, lambda trace: _first_lead(trace)),
        lowest_prediction_lead_case_id=_min_case(trace_tuple, lambda trace: _first_lead(trace)),
        maximum_adaptive_restitution_error_case_id=_max_case(benchmark.comparisons, lambda item: item.adaptive_restitution_error),
        maximum_adaptive_penetration_error_case_id=_max_case(benchmark.comparisons, lambda item: item.adaptive_penetration_error),
        least_converged_reference_case_id=_least_converged_case(convergence_tuple),
    )


def build_attribution_dataset(
    *,
    cases: Sequence[ContactBenchmarkCase],
    benchmark: ContactBenchmarkDataset,
    selection: ConvergenceSelectionConfig = ConvergenceSelectionConfig(),
    convergence_config: ReferenceConvergenceConfig = ReferenceConvergenceConfig(),
    validation: BenchmarkValidationConfig = BenchmarkValidationConfig(),
    recommendation: SubstepRecommendationConfig = SubstepRecommendationConfig(maximum_substeps=16),
) -> AttributionDiagnosticDataset:
    """Select cases, run convergence and trace diagnostics, and attribute failures."""
    selected_cases = select_convergence_cases(cases, benchmark, config=selection)
    convergence = tuple(
        run_reference_convergence(case, recommendation=recommendation, validation=validation, config=convergence_config)
        for case in selected_cases
    )
    traces = tuple(run_adaptive_diagnostic_trace(case, recommendation=recommendation) for case in selected_cases)
    by_comparison = {item.case_id: item for item in benchmark.comparisons}
    by_adaptive = {
        result.case_id: result for result in benchmark.results if result.mode is BenchmarkMode.ADAPTIVE
    }
    by_convergence = {item.case_id: item for item in convergence}
    by_trace = {item.case_id: item for item in traces}
    attributions = tuple(
        attribute_adaptive_failure(
            comparison=by_comparison[case.case_id],
            adaptive_result=by_adaptive[case.case_id],
            trace=by_trace[case.case_id],
            convergence=by_convergence.get(case.case_id),
            recommendation=recommendation,
        )
        for case in selected_cases
    )
    summary = build_attribution_summary(
        benchmark=benchmark,
        convergence=convergence,
        traces=traces,
        attributions=attributions,
    )
    return AttributionDiagnosticDataset(
        benchmark=benchmark,
        convergence=convergence,
        traces=traces,
        attributions=attributions,
        summary=summary,
        config={
            "selection": asdict(selection),
            "convergence": asdict(convergence_config),
            "validation": asdict(validation),
            "recommendation": asdict(recommendation),
        },
        units=_units(),
    )


def export_convergence_csv(results: Sequence[ReferenceConvergenceResult], path: str | Path) -> None:
    """Export reference convergence levels as CSV."""
    rows = []
    for result in results:
        for level in result.levels:
            rows.append({
                "case_id": result.case_id,
                "overall_status": result.overall_status.value,
                **_level_row(level),
            })
    _write_csv(rows, path, ("case_id", "overall_status", *(_level_row(_empty_level()).keys())))


def export_attribution_csv(attributions: Sequence[AdaptiveFailureAttribution], path: str | Path) -> None:
    """Export adaptive attribution rows as CSV."""
    rows = [
        {
            "case_id": item.case_id,
            "primary_reason": item.primary_reason.value,
            "secondary_reasons": ";".join(reason.value for reason in item.secondary_reasons),
            "restitution_outcome": item.restitution_outcome.value,
            "penetration_outcome": item.penetration_outcome.value,
            "evidence": " | ".join(item.evidence),
        }
        for item in attributions
    ]
    _write_csv(rows, path, ("case_id", "primary_reason", "secondary_reasons", "restitution_outcome", "penetration_outcome", "evidence"))


def export_diagnostic_json(dataset: AttributionDiagnosticDataset, path: str | Path) -> None:
    """Export the full attribution dataset as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_diagnostic_dataset_to_dict(dataset), indent=2, sort_keys=True), encoding="utf8")


def write_attribution_markdown_report(dataset: AttributionDiagnosticDataset, path: str | Path) -> None:
    """Write a Markdown attribution report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_attribution_markdown_report(dataset), encoding="utf8")


def build_attribution_markdown_report(dataset: AttributionDiagnosticDataset) -> str:
    """Build a Markdown report for convergence and attribution diagnostics."""
    summary = dataset.summary
    lines = [
        "# Adaptive Failure Attribution",
        "",
        "## Overview",
        "",
        f"- total cases: {summary.total_cases}",
        f"- convergence checked cases: {summary.convergence_checked_cases}",
        f"- reference converged / unresolved: {summary.converged_reference_cases} / {summary.unresolved_reference_cases}",
        f"- failure reason counts: {dict(summary.failure_reason_counts)}",
        f"- improvement outcome counts: {dict(summary.improvement_outcome_counts)}",
        f"- mean prediction lead macro steps: {_fmt(summary.mean_prediction_lead_macro_steps)}",
        f"- minimum prediction lead macro steps: {_fmt(summary.minimum_prediction_lead_macro_steps)}",
        "",
        "## Reference Convergence",
        "",
        "| case | status | metric | D1 | D2 | ratio |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for result in dataset.convergence:
        for metric in (result.restitution, result.rebound_speed, result.maximum_penetration, result.contact_duration):
            lines.append(
                f"| {result.case_id} | {result.overall_status.value} | {metric.metric_name} | "
                f"{_fmt(metric.coarse_to_fine_difference)} | {_fmt(metric.fine_to_finer_difference)} | {_fmt(metric.difference_ratio)} |"
            )
    lines.extend(["", "## Failure Attribution", ""])
    grouped: dict[str, list[AdaptiveFailureAttribution]] = {}
    for attribution in dataset.attributions:
        grouped.setdefault(attribution.primary_reason.value, []).append(attribution)
    for reason in sorted(grouped):
        lines.append(f"### {reason}")
        for item in sorted(grouped[reason], key=lambda value: value.case_id):
            lines.append(f"- {item.case_id}: {'; '.join(item.evidence) or 'no diagnostic evidence'}")
        lines.append("")
    lines.extend([
        "## Worst Cases",
        "",
        f"- max restitution error: {summary.maximum_adaptive_restitution_error_case_id or 'none'}",
        f"- max penetration error: {summary.maximum_adaptive_penetration_error_case_id or 'none'}",
        f"- shortest prediction lead: {summary.lowest_prediction_lead_case_id or 'none'}",
        f"- most contact episodes: {_most_episodes_case(dataset.traces) or 'none'}",
        f"- least converged reference: {summary.least_converged_reference_case_id or 'none'}",
        "",
        "## Conclusion",
        "",
        _conclusion(summary),
    ])
    return "\n".join(lines) + "\n"


class _EpisodeBuilder:
    def __init__(self, *, episode_index: int, candidate_id: str, macro_timestep: float) -> None:
        self.episode_index = episode_index
        self.candidate_id = candidate_id
        self.macro_timestep = macro_timestep
        self.prediction_time: float | None = None
        self.predicted_absolute_contact_time: float | None = None
        self.first_actual_contact_time: float | None = None
        self.approaching_start_time: float | None = None
        self.impacting_start_time: float | None = None
        self.separating_start_time: float | None = None
        self.resting_start_time: float | None = None
        self.episode_end_time: float | None = None
        self.gap_at_first_prediction: float | None = None
        self.approach_speed_at_first_prediction: float | None = None
        self.solver_characteristic_timescale: float | None = None
        self.requested_substep_count: int | None = None
        self.maximum_actual_substep_count = 1
        self.limited_by_maximum_substeps = False
        self.minimum_actual_timestep = macro_timestep
        self.substepped_macro_step_count = 0
        self.contact_substep_count = 0
        self.maximum_penetration = 0.0
        self.maximum_penetration_time: float | None = None
        self.state_at_maximum_penetration: ContactMotionState | None = None
        self.first_contact_end_time: float | None = None

    def observe_decision(self, decision, macro_start: float, macro_end: float) -> None:
        self.maximum_actual_substep_count = max(self.maximum_actual_substep_count, decision.substep_count)
        self.minimum_actual_timestep = min(self.minimum_actual_timestep, decision.actual_substep_timestep)
        if decision.substep_count > 1:
            self.substepped_macro_step_count += 1
        if decision.state_after is ContactMotionState.APPROACHING and self.approaching_start_time is None:
            self.approaching_start_time = macro_end
        if decision.state_after is ContactMotionState.IMPACTING and self.impacting_start_time is None:
            self.impacting_start_time = macro_end
        if decision.state_after is ContactMotionState.SEPARATING and self.separating_start_time is None:
            self.separating_start_time = macro_end
        if decision.state_after is ContactMotionState.RESTING and self.resting_start_time is None:
            self.resting_start_time = macro_end
        if decision.prediction is not None and self.prediction_time is None:
            self.prediction_time = macro_start
            self.predicted_absolute_contact_time = macro_start + decision.prediction.time_to_contact
            self.gap_at_first_prediction = decision.prediction.gap
            self.approach_speed_at_first_prediction = decision.prediction.normal_approach_speed
        if decision.solver_estimate is not None:
            self.solver_characteristic_timescale = decision.solver_estimate.timescale.characteristic_timescale
            if self.requested_substep_count is None:
                self.requested_substep_count = decision.solver_estimate.recommendation.substep_count
            self.limited_by_maximum_substeps = (
                self.limited_by_maximum_substeps
                or decision.solver_estimate.recommendation.limited_by_maximum_substeps
            )

    def observe_samples(
        self,
        case: ContactBenchmarkCase,
        samples: Sequence,
        state: ContactMotionState,
    ) -> None:
        for sample in samples:
            contacts = _benchmark._contacts_for_case(case, sample)
            if contacts:
                if self.first_actual_contact_time is None:
                    self.first_actual_contact_time = sample.time
                self.first_contact_end_time = sample.time
                self.contact_substep_count += 1
                penetration = max(contact.penetration_depth for contact in contacts)
                if penetration > self.maximum_penetration:
                    self.maximum_penetration = penetration
                    self.maximum_penetration_time = sample.time
                    self.state_at_maximum_penetration = state

    def build(self) -> AdaptiveContactEpisodeTrace:
        lead_seconds = (
            None
            if self.prediction_time is None or self.first_actual_contact_time is None
            else self.first_actual_contact_time - self.prediction_time
        )
        return AdaptiveContactEpisodeTrace(
            episode_index=self.episode_index,
            candidate_id=self.candidate_id,
            prediction_time=self.prediction_time,
            predicted_absolute_contact_time=self.predicted_absolute_contact_time,
            first_actual_contact_time=self.first_actual_contact_time,
            prediction_lead_time_seconds=lead_seconds,
            prediction_lead_time_macro_steps=None if lead_seconds is None else lead_seconds / self.macro_timestep,
            approaching_start_time=self.approaching_start_time,
            impacting_start_time=self.impacting_start_time,
            separating_start_time=self.separating_start_time,
            resting_start_time=self.resting_start_time,
            episode_end_time=self.episode_end_time,
            gap_at_first_prediction=self.gap_at_first_prediction,
            approach_speed_at_first_prediction=self.approach_speed_at_first_prediction,
            solver_characteristic_timescale=self.solver_characteristic_timescale,
            requested_substep_count=self.requested_substep_count,
            maximum_actual_substep_count=self.maximum_actual_substep_count,
            limited_by_maximum_substeps=self.limited_by_maximum_substeps,
            minimum_actual_timestep=self.minimum_actual_timestep,
            substepped_macro_step_count=self.substepped_macro_step_count,
            contact_substep_count=self.contact_substep_count,
            maximum_penetration=self.maximum_penetration,
            maximum_penetration_time=self.maximum_penetration_time,
            state_at_maximum_penetration=self.state_at_maximum_penetration,
            contact_duration_seconds=None
            if self.first_actual_contact_time is None or self.first_contact_end_time is None
            else self.first_contact_end_time - self.first_actual_contact_time,
        )


def _metric_convergence(
    metric_name: str,
    values: Sequence[float | None],
    absolute_tolerance: float,
    relative_tolerance: float,
    *,
    invalid: bool,
    required: bool = True,
) -> ReferenceMetricConvergence:
    if invalid:
        status = ReferenceConvergenceStatus.INVALID_RESULT
        return ReferenceMetricConvergence(metric_name, None, None, None, absolute_tolerance, relative_tolerance, status)
    if len(values) < 3:
        status = ReferenceConvergenceStatus.INSUFFICIENT_LEVELS
        return ReferenceMetricConvergence(metric_name, None, None, None, absolute_tolerance, relative_tolerance, status)
    if any(value is None or _bad(value) for value in values[-3:]):
        status = ReferenceConvergenceStatus.INVALID_RESULT if required else ReferenceConvergenceStatus.NOT_CONVERGED
        return ReferenceMetricConvergence(metric_name, None, None, None, absolute_tolerance, relative_tolerance, status)
    q0, q1, q2 = values[-3:]  # type: ignore[misc]
    d1 = abs(q0 - q1)
    d2 = abs(q1 - q2)
    ratio = None if d1 <= 1.0e-15 else d2 / d1
    scale = max(abs(q2), abs(q1), 1.0e-12)
    tolerance_met = d2 <= absolute_tolerance or d2 <= relative_tolerance * scale
    non_increasing = True if d1 <= 1.0e-15 else d2 <= d1
    status = ReferenceConvergenceStatus.CONVERGED if tolerance_met and non_increasing else ReferenceConvergenceStatus.NOT_CONVERGED
    return ReferenceMetricConvergence(metric_name, d1, d2, ratio, absolute_tolerance, relative_tolerance, status)


def _improvement_outcome(
    coarse_error: float | None,
    adaptive_error: float | None,
    tolerance: float,
    convergence: ReferenceConvergenceResult | None,
) -> ImprovementOutcome:
    if convergence is not None and convergence.overall_status is not ReferenceConvergenceStatus.CONVERGED:
        return ImprovementOutcome.REFERENCE_UNRESOLVED
    if coarse_error is None or adaptive_error is None:
        return ImprovementOutcome.NOT_APPLICABLE
    if coarse_error <= tolerance and adaptive_error <= tolerance:
        return ImprovementOutcome.BOTH_ACCEPTABLE
    return ImprovementOutcome.IMPROVED if adaptive_error <= coarse_error else ImprovementOutcome.NOT_IMPROVED


def _primary_reason(reasons: Sequence[AdaptiveFailureReason]) -> AdaptiveFailureReason:
    priority = (
        AdaptiveFailureReason.NONPHYSICAL_ADAPTIVE_RESULT,
        AdaptiveFailureReason.REFERENCE_NOT_CONVERGED,
        AdaptiveFailureReason.LATE_PREDICTION,
        AdaptiveFailureReason.SHORT_PREDICTION_LEAD,
        AdaptiveFailureReason.MAX_SUBSTEPS_LIMITED,
        AdaptiveFailureReason.INSUFFICIENT_TIME_RESOLUTION,
        AdaptiveFailureReason.EARLY_FINE_EXIT,
        AdaptiveFailureReason.MULTIPLE_CONTACT_EPISODES,
        AdaptiveFailureReason.METRIC_SAMPLING_SENSITIVITY,
    )
    for reason in priority:
        if reason in reasons:
            return reason
    return AdaptiveFailureReason.NONE


def _dedupe_reasons(reasons: Sequence[AdaptiveFailureReason]) -> list[AdaptiveFailureReason]:
    deduped: list[AdaptiveFailureReason] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped


def _first_lead(trace: AdaptiveDiagnosticTrace) -> float | None:
    leads = [episode.prediction_lead_time_macro_steps for episode in trace.episodes if episode.prediction_lead_time_macro_steps is not None]
    return None if not leads else leads[0]


def _insufficient_resolution(trace: AdaptiveDiagnosticTrace, recommendation: SubstepRecommendationConfig) -> bool:
    for episode in trace.episodes:
        if episode.limited_by_maximum_substeps:
            continue
        if episode.solver_characteristic_timescale is None:
            continue
        samples = episode.solver_characteristic_timescale / episode.minimum_actual_timestep
        if samples < recommendation.samples_per_characteristic_time:
            return True
    return False


def _early_fine_exit(trace: AdaptiveDiagnosticTrace) -> bool:
    fine_states = {
        ContactMotionState.APPROACHING,
        ContactMotionState.IMPACTING,
        ContactMotionState.SEPARATING,
    }
    return any(
        episode.maximum_penetration > 0.0
        and episode.state_at_maximum_penetration is not None
        and episode.state_at_maximum_penetration not in fine_states
        and episode.maximum_actual_substep_count <= 1
        for episode in trace.episodes
    )


def _metric_sampling_sensitive(convergence: ReferenceConvergenceResult | None) -> bool:
    if convergence is None:
        return False
    metric = convergence.maximum_penetration
    return (
        metric.coarse_to_fine_difference is not None
        and metric.fine_to_finer_difference is not None
        and metric.fine_to_finer_difference > metric.coarse_to_fine_difference
        and convergence.rebound_speed.status is ReferenceConvergenceStatus.CONVERGED
    )


def _top_k(comparisons: Sequence[BenchmarkComparison], k: int, selector) -> tuple[str, ...]:
    if k <= 0:
        return ()
    scored = [
        (selector(item), item.case_id)
        for item in comparisons
        if selector(item) is not None
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(case_id for _, case_id in scored[:k])


def _level_row(level: ReferenceLevelResult) -> dict[str, object]:
    return {
        "refinement_level": level.refinement_level,
        "timestep": level.timestep,
        "refinement_factor": level.refinement_factor,
        "restitution": level.restitution,
        "rebound_speed": level.rebound_speed,
        "maximum_penetration": level.maximum_penetration,
        "contact_duration_seconds": level.contact_duration_seconds,
        "validity": level.validity.value,
        "physics_step_count": level.physics_step_count,
    }


def _empty_level() -> ReferenceLevelResult:
    return ReferenceLevelResult(0, 0.0, 1, None, None, 0.0, None, BenchmarkValidity.UNSTABLE, 0)


def _write_csv(rows: Sequence[dict[str, object]], path: str | Path, fieldnames: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _diagnostic_dataset_to_dict(dataset: AttributionDiagnosticDataset) -> dict[str, object]:
    return {
        "config": dataset.config,
        "mujoco_version": dataset.benchmark.mujoco_version,
        "benchmark": {
            "cases": list(dataset.benchmark.cases),
            "comparisons": [asdict(item) for item in dataset.benchmark.comparisons],
        },
        "reference_convergence": [_convergence_to_dict(item) for item in dataset.convergence],
        "adaptive_traces": [_trace_to_dict(item) for item in dataset.traces],
        "failure_attribution": [_attribution_to_dict(item) for item in dataset.attributions],
        "summary": asdict(dataset.summary),
        "units": dataset.units,
    }


def _convergence_to_dict(result: ReferenceConvergenceResult) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "levels": [_level_row(level) for level in result.levels],
        "restitution": _metric_to_dict(result.restitution),
        "rebound_speed": _metric_to_dict(result.rebound_speed),
        "maximum_penetration": _metric_to_dict(result.maximum_penetration),
        "contact_duration": _metric_to_dict(result.contact_duration),
        "overall_status": result.overall_status.value,
        "selected_reference_level": result.selected_reference_level,
    }


def _metric_to_dict(metric: ReferenceMetricConvergence) -> dict[str, object]:
    data = asdict(metric)
    data["status"] = metric.status.value
    return data


def _trace_to_dict(trace: AdaptiveDiagnosticTrace) -> dict[str, object]:
    data = asdict(trace)
    for episode in data["episodes"]:
        if episode["state_at_maximum_penetration"] is not None:
            episode["state_at_maximum_penetration"] = episode["state_at_maximum_penetration"].value
    return data


def _attribution_to_dict(attribution: AdaptiveFailureAttribution) -> dict[str, object]:
    data = asdict(attribution)
    data["primary_reason"] = attribution.primary_reason.value
    data["secondary_reasons"] = [reason.value for reason in attribution.secondary_reasons]
    data["restitution_outcome"] = attribution.restitution_outcome.value
    data["penetration_outcome"] = attribution.penetration_outcome.value
    return data


def _units() -> dict[str, str]:
    return {
        "timestep": "s",
        "prediction_lead_time_seconds": "s",
        "prediction_lead_time_macro_steps": "macro steps",
        "maximum_penetration": "m",
        "rebound_speed": "m/s",
        "contact_duration_seconds": "s",
    }


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.6g}"


def _bad(value: float | None) -> bool:
    return value is not None and not math.isfinite(value)


def _max_case(items: Sequence, selector) -> str | None:
    scored = [(selector(item), item.case_id) for item in items if selector(item) is not None]
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _min_case(items: Sequence, selector) -> str | None:
    scored = [(selector(item), item.case_id) for item in items if selector(item) is not None]
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][1]


def _least_converged_case(results: Sequence[ReferenceConvergenceResult]) -> str | None:
    scored = [
        (
            max(
                value for value in (
                    result.restitution.difference_ratio,
                    result.maximum_penetration.difference_ratio,
                )
                if value is not None
            ),
            result.case_id,
        )
        for result in results
        if result.restitution.difference_ratio is not None or result.maximum_penetration.difference_ratio is not None
    ]
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _most_episodes_case(traces: Sequence[AdaptiveDiagnosticTrace]) -> str | None:
    if not traces:
        return None
    return max(traces, key=lambda item: (item.total_contact_episode_count, item.case_id)).case_id


def _conclusion(summary: AttributionBenchmarkSummary) -> str:
    if not summary.failure_reason_counts:
        return "No attribution cases were selected."
    dominant, count = max(summary.failure_reason_counts.items(), key=lambda item: (item[1], item[0]))
    return f"The most frequent diagnostic reason is `{dominant}` with {count} occurrence(s)."
