"""Unified batch evaluation for adaptive primary-impact diagnostics."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from physical_simulation.evaluation.contact_benchmark import (
    BenchmarkComparison,
    BenchmarkMode,
    BenchmarkValidationConfig,
    BenchmarkValidity,
    ContactBenchmarkResult,
    SpherePlaneBenchmarkCase,
    SphereSphereBenchmarkCase,
    compare_benchmark_results,
    run_contact_benchmark,
)
from physical_simulation.evaluation.contact_convergence import (
    AdaptiveDiagnosticTrace,
    AdaptiveFailureReason,
    ImprovementOutcome,
    ReferenceConvergenceConfig,
    ReferenceConvergenceResult,
    ReferenceConvergenceStatus,
    run_adaptive_diagnostic_trace,
    run_reference_convergence,
)
from physical_simulation.evaluation.contact_episode import (
    ContactEpisodeMetrics,
    ContactEpisodeSegmentationConfig,
    EpisodeMatch,
    EpisodeMatchingConfig,
    EpisodeMatchStatus,
    EpisodeReferenceConvergenceResult,
    build_primary_impact_comparison,
    collect_contact_episode_samples,
    match_contact_episodes,
    run_episode_reference_convergence,
    segment_contact_episodes,
)
from physical_simulation.evaluation.primary_attribution import (
    AdaptiveAttributionConfig,
    AttributionScope,
    PrimaryImpactAttributionInput,
    PrimaryImpactCaseOutcome,
    PrimaryImpactFailureAttribution,
    PrimaryImpactImprovementConfig,
    attribute_primary_impact_failure,
    primary_improvement_rates,
)
from physical_simulation.mujoco import (
    AdaptiveSubstepConfig,
    AnalyticPlane,
    MuJoCoContactSolverParams,
    SubstepRecommendationConfig,
)
from physical_simulation.validation.asset_validator import _finite_float
from physical_simulation.validation.errors import PhysicsValidationError

Vector3 = tuple[float, float, float]


class AdaptiveBatchSceneType(Enum):
    """Scene families supported by the batch evaluator."""

    SPHERE_PLANE = "sphere_plane"
    SPHERE_SPHERE = "sphere_sphere"


class ReferenceEvaluationStatus(Enum):
    """Batch-level distinction between unchecked and unresolved references."""

    NOT_CHECKED = "not_checked"
    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    INVALID = "invalid"


class ReferenceEvaluationMode(Enum):
    """Strategy for expensive reference refinement."""

    ALL = "all"
    SELECTED = "selected"
    NONE = "none"


@dataclass(frozen=True)
class BatchGenerationConfig:
    """Deterministic case sampling configuration."""

    include_full_cartesian_product: bool = False
    maximum_case_count: int | None = None
    sampling_seed: int = 0

    def __post_init__(self) -> None:
        if self.maximum_case_count is not None and self.maximum_case_count < 1:
            raise PhysicsValidationError("maximum_case_count must be positive or None")
        if not isinstance(self.sampling_seed, int) or isinstance(self.sampling_seed, bool):
            raise PhysicsValidationError("sampling_seed must be an int")


@dataclass(frozen=True)
class BatchReferenceConfig:
    """Configure selected reference convergence checks."""

    mode: ReferenceEvaluationMode = ReferenceEvaluationMode.SELECTED
    refinement_factors: tuple[int, ...] = (1, 2, 4)
    include_all_nonphysical_coarse: bool = True
    include_all_primary_not_improved: bool = True
    top_k_adaptive_restitution_error: int = 5
    top_k_adaptive_penetration_error: int = 5
    top_k_adaptive_duration_error: int = 5
    maximum_selected_cases: int | None = 20

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ReferenceEvaluationMode):
            raise PhysicsValidationError("mode must be a ReferenceEvaluationMode")
        if len(self.refinement_factors) < 1 or any(f < 1 for f in self.refinement_factors):
            raise PhysicsValidationError("refinement_factors must contain positive ints")
        object.__setattr__(self, "refinement_factors", tuple(sorted(set(self.refinement_factors))))
        for field_name in (
            "top_k_adaptive_restitution_error",
            "top_k_adaptive_penetration_error",
            "top_k_adaptive_duration_error",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PhysicsValidationError(f"{field_name} must be a non-negative int")
        if self.maximum_selected_cases is not None and self.maximum_selected_cases < 1:
            raise PhysicsValidationError("maximum_selected_cases must be positive or None")


@dataclass(frozen=True)
class AdaptiveBatchConfig:
    """Configuration for the unified batch pipeline."""

    reference: BatchReferenceConfig = field(default_factory=BatchReferenceConfig)
    benchmark_validation: BenchmarkValidationConfig = field(default_factory=BenchmarkValidationConfig)
    episode_segmentation: ContactEpisodeSegmentationConfig = field(
        default_factory=lambda: ContactEpisodeSegmentationConfig(maximum_chatter_gap_seconds=1.0 / 240.0 / 16.0 * 2.0)
    )
    episode_matching: EpisodeMatchingConfig = field(default_factory=lambda: EpisodeMatchingConfig(maximum_start_time_difference=1.0 / 120.0))
    improvement: PrimaryImpactImprovementConfig = field(default_factory=PrimaryImpactImprovementConfig)
    attribution: AdaptiveAttributionConfig = field(default_factory=AdaptiveAttributionConfig)
    fail_fast: bool = False
    collect_substep_samples: bool = True
    export_csv: bool = True
    export_json: bool = True
    export_markdown: bool = True
    output_dir: str | Path = Path("artifacts/adaptive_primary_batch")
    recommendation: SubstepRecommendationConfig = field(default_factory=lambda: SubstepRecommendationConfig(maximum_substeps=16))


@dataclass(frozen=True)
class AdaptiveBatchCase:
    """One declarative batch case."""

    case_id: str
    scene_type: AdaptiveBatchSceneType
    macro_timestep: float
    total_simulation_time: float
    contact_params: MuJoCoContactSolverParams
    adaptive_config: AdaptiveSubstepConfig
    sphere_a_radius: float
    sphere_a_mass: float
    sphere_a_initial_position: Vector3
    sphere_a_initial_velocity: Vector3
    sphere_b_radius: float | None = None
    sphere_b_mass: float | None = None
    sphere_b_initial_position: Vector3 | None = None
    sphere_b_initial_velocity: Vector3 | None = None
    plane: AnalyticPlane | None = None
    metadata: Mapping[str, str | int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise PhysicsValidationError("case_id must be non-empty")
        if not isinstance(self.scene_type, AdaptiveBatchSceneType):
            raise PhysicsValidationError("scene_type must be an AdaptiveBatchSceneType")
        for field_name in ("macro_timestep", "total_simulation_time", "sphere_a_radius", "sphere_a_mass"):
            minimum = 0.0
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), field_name=field_name, minimum=minimum, strict_minimum=True),
            )
        object.__setattr__(self, "sphere_a_initial_position", _vector3(self.sphere_a_initial_position, "sphere_a_initial_position"))
        object.__setattr__(self, "sphere_a_initial_velocity", _vector3(self.sphere_a_initial_velocity, "sphere_a_initial_velocity"))
        if self.scene_type is AdaptiveBatchSceneType.SPHERE_PLANE:
            if self.plane is None:
                raise PhysicsValidationError("sphere-plane cases must provide plane")
        else:
            if (
                self.sphere_b_radius is None
                or self.sphere_b_mass is None
                or self.sphere_b_initial_position is None
                or self.sphere_b_initial_velocity is None
            ):
                raise PhysicsValidationError("sphere-sphere cases must provide sphere_b parameters")
            object.__setattr__(self, "sphere_b_radius", _finite_float(self.sphere_b_radius, field_name="sphere_b_radius", minimum=0.0, strict_minimum=True))
            object.__setattr__(self, "sphere_b_mass", _finite_float(self.sphere_b_mass, field_name="sphere_b_mass", minimum=0.0, strict_minimum=True))
            object.__setattr__(self, "sphere_b_initial_position", _vector3(self.sphere_b_initial_position, "sphere_b_initial_position"))
            object.__setattr__(self, "sphere_b_initial_velocity", _vector3(self.sphere_b_initial_velocity, "sphere_b_initial_velocity"))
        for key, value in self.metadata.items():
            if not isinstance(key, str) or not isinstance(value, (str, int, float)) or isinstance(value, bool):
                raise PhysicsValidationError("metadata must map strings to str/int/float values")
            if isinstance(value, float) and not math.isfinite(value):
                raise PhysicsValidationError("metadata float values must be finite")


@dataclass(frozen=True)
class AdaptiveBatchCaseResult:
    """Complete diagnostic result for one batch case."""

    case: AdaptiveBatchCase
    coarse_run: ContactBenchmarkResult | None
    fine_run: ContactBenchmarkResult | None
    adaptive_run: ContactBenchmarkResult | None
    coarse_episodes: tuple[ContactEpisodeMetrics, ...]
    fine_episodes: tuple[ContactEpisodeMetrics, ...]
    adaptive_episodes: tuple[ContactEpisodeMetrics, ...]
    coarse_primary_match: EpisodeMatch | None
    adaptive_primary_match: EpisodeMatch | None
    provisional_primary_comparison: object | None
    reference_evaluation_status: ReferenceEvaluationStatus
    primary_reference_convergence: EpisodeReferenceConvergenceResult | None
    converged_reference_episode: ContactEpisodeMetrics | None
    primary_attribution: PrimaryImpactFailureAttribution | None
    adaptive_trace: AdaptiveDiagnosticTrace | None
    run_level_comparison: BenchmarkComparison | None
    run_level_reference_convergence: ReferenceConvergenceResult | None
    error: str | None


@dataclass(frozen=True)
class AdaptiveBatchSummary:
    """Aggregated batch metrics."""

    total_case_count: int
    completed_case_count: int
    invalid_case_count: int
    sphere_plane_case_count: int
    sphere_sphere_case_count: int
    primary_matched_case_count: int
    primary_unmatched_case_count: int
    reference_checked_case_count: int
    reference_not_checked_case_count: int
    reference_converged_case_count: int
    reference_unresolved_case_count: int
    primary_scope_case_count: int
    fallback_scope_case_count: int
    unavailable_scope_case_count: int
    primary_case_outcome_counts: Mapping[str, int]
    primary_reason_counts: Mapping[str, int]
    restitution_outcome_counts: Mapping[str, int]
    penetration_outcome_counts: Mapping[str, int]
    duration_outcome_counts: Mapping[str, int]
    primary_restitution_improvement_rate: float | None
    primary_penetration_improvement_rate: float | None
    primary_duration_improvement_rate: float | None
    primary_case_improvement_rate: float | None
    primary_restitution_improvement_numerator: int
    primary_restitution_improvement_denominator: int
    primary_penetration_improvement_numerator: int
    primary_penetration_improvement_denominator: int
    primary_duration_improvement_numerator: int
    primary_duration_improvement_denominator: int
    primary_case_improvement_numerator: int
    primary_case_improvement_denominator: int
    mean_adaptive_step_ratio: float | None
    median_adaptive_step_ratio: float | None
    maximum_adaptive_step_ratio: float | None
    mean_adaptive_step_saving: float | None
    median_adaptive_step_saving: float | None
    mean_primary_restitution_error: float | None
    maximum_primary_restitution_error: float | None
    mean_primary_penetration_error: float | None
    maximum_primary_penetration_error: float | None
    mean_primary_duration_error: float | None
    maximum_primary_duration_error: float | None


@dataclass(frozen=True)
class AdaptiveBatchGroupSummary:
    """Grouped batch summary."""

    group_name: str
    group_value: str
    case_count: int
    reference_converged_count: int
    restitution_improvement_rate: float | None
    penetration_improvement_rate: float | None
    duration_improvement_rate: float | None
    mean_step_saving: float | None
    mean_restitution_error: float | None
    mean_penetration_error: float | None


@dataclass(frozen=True)
class AccuracyCostPoint:
    """One converged case represented in accuracy/cost space."""

    case_id: str
    scene_type: AdaptiveBatchSceneType
    restitution_error: float | None
    penetration_error: float | None
    duration_error: float | None
    adaptive_step_ratio: float
    adaptive_step_saving: float
    maximum_substep_count: int
    substepped_macro_step_ratio: float


@dataclass(frozen=True)
class AdaptiveBatchWorstCases:
    """Stable worst-case identifiers."""

    maximum_primary_restitution_error_case_id: str | None
    maximum_primary_penetration_error_case_id: str | None
    maximum_primary_duration_error_case_id: str | None
    maximum_adaptive_step_ratio_case_id: str | None
    shortest_prediction_lead_case_id: str | None
    maximum_substep_limited_case_id: str | None
    reference_least_converged_case_id: str | None
    primary_episode_unmatched_case_id: str | None
    adaptive_nonphysical_case_id: str | None


@dataclass(frozen=True)
class AdaptiveBatchResult:
    """Complete batch result and exported artifact paths."""

    cases: tuple[AdaptiveBatchCase, ...]
    results: tuple[AdaptiveBatchCaseResult, ...]
    selected_reference_case_ids: tuple[str, ...]
    summary: AdaptiveBatchSummary
    group_summaries: tuple[AdaptiveBatchGroupSummary, ...]
    accuracy_cost_points: tuple[AccuracyCostPoint, ...]
    nondominated_accuracy_cost_points: tuple[AccuracyCostPoint, ...]
    worst_cases: AdaptiveBatchWorstCases
    artifact_paths: Mapping[str, str]
    config: AdaptiveBatchConfig
    mujoco_version: str | None
    git_commit: str | None


def generate_sphere_plane_batch_cases(
    *,
    config: BatchGenerationConfig = BatchGenerationConfig(),
    adaptive_config: AdaptiveSubstepConfig = AdaptiveSubstepConfig(),
) -> tuple[AdaptiveBatchCase, ...]:
    """Generate deterministic sphere-plane batch cases."""
    heights = (0.4, 0.7, 1.0, 1.3)
    timesteps = (1.0 / 120.0, 1.0 / 240.0, 1.0 / 480.0)
    solrefs = ((0.02, 0.3), (0.02, 0.5), (0.01, 0.3))
    radii = (0.05, 0.10)
    masses = (0.5, 1.0, 2.0)
    combos = [
        (height, timestep, solref, radius, mass)
        for height in heights
        for timestep in timesteps
        for solref in solrefs
        for radius in radii
        for mass in masses
    ]
    selected = combos if config.include_full_cartesian_product else _deterministic_layered_sample(combos, config, target_count=24)
    if config.maximum_case_count is not None:
        selected = selected[: config.maximum_case_count]
    return tuple(_sphere_plane_case(*combo, adaptive_config=adaptive_config) for combo in selected)


def generate_sphere_sphere_batch_cases(
    *,
    config: BatchGenerationConfig = BatchGenerationConfig(maximum_case_count=8),
    adaptive_config: AdaptiveSubstepConfig = AdaptiveSubstepConfig(),
) -> tuple[AdaptiveBatchCase, ...]:
    """Generate deterministic sphere-sphere batch cases."""
    base = (
        ("symmetric_equal_mass", 1.0, 1.0, 1.5, -1.5, 0.1, (0.01, 0.3)),
        ("different_mass", 0.5, 2.0, 1.5, -1.5, 0.1, (0.01, 0.3)),
        ("one_static_initially", 1.0, 1.0, 2.0, 0.0, 0.1, (0.02, 0.3)),
        ("slow_relative", 1.0, 1.0, 0.75, -0.75, 0.1, (0.02, 0.5)),
        ("fast_relative", 1.0, 1.0, 2.5, -2.5, 0.1, (0.01, 0.3)),
        ("small_radius", 1.0, 1.0, 1.5, -1.5, 0.05, (0.02, 0.3)),
    )
    selected = list(base)
    if config.maximum_case_count is not None:
        selected = selected[: config.maximum_case_count]
    return tuple(_sphere_sphere_case(*item, adaptive_config=adaptive_config) for item in selected)


def make_smoke_adaptive_batch() -> tuple[tuple[AdaptiveBatchCase, ...], AdaptiveBatchConfig]:
    """Create a fast 6-10 case batch with at least two reference checks."""
    adaptive = AdaptiveSubstepConfig()
    cases = (
        *generate_sphere_plane_batch_cases(config=BatchGenerationConfig(maximum_case_count=6), adaptive_config=adaptive),
        *generate_sphere_sphere_batch_cases(config=BatchGenerationConfig(maximum_case_count=2), adaptive_config=adaptive),
    )
    config = AdaptiveBatchConfig(
        reference=BatchReferenceConfig(mode=ReferenceEvaluationMode.SELECTED, maximum_selected_cases=4, top_k_adaptive_restitution_error=2, top_k_adaptive_penetration_error=2),
        episode_matching=EpisodeMatchingConfig(maximum_start_time_difference=1.0 / 120.0),
    )
    return tuple(cases), config


def make_standard_adaptive_batch() -> tuple[tuple[AdaptiveBatchCase, ...], AdaptiveBatchConfig]:
    """Create an approximately 30-60 case standard batch."""
    adaptive = AdaptiveSubstepConfig()
    cases = (
        *generate_sphere_plane_batch_cases(config=BatchGenerationConfig(maximum_case_count=36), adaptive_config=adaptive),
        *generate_sphere_sphere_batch_cases(config=BatchGenerationConfig(maximum_case_count=6), adaptive_config=adaptive),
    )
    config = AdaptiveBatchConfig(
        reference=BatchReferenceConfig(mode=ReferenceEvaluationMode.SELECTED, maximum_selected_cases=20),
        episode_matching=EpisodeMatchingConfig(maximum_start_time_difference=1.0 / 120.0),
    )
    return tuple(cases), config


def run_adaptive_primary_batch(
    cases: Sequence[AdaptiveBatchCase],
    *,
    config: AdaptiveBatchConfig,
) -> AdaptiveBatchResult:
    """Run the complete deterministic primary-impact batch pipeline."""
    case_tuple = _validate_unique_cases(cases)
    provisional: list[AdaptiveBatchCaseResult] = []
    for case in case_tuple:
        try:
            provisional.append(_run_provisional_case(case, config))
        except Exception as exc:  # noqa: BLE001 - batch isolation is intentional.
            if config.fail_fast:
                raise
            provisional.append(_invalid_case_result(case, str(exc)))
    selected_ids = _select_reference_case_ids(provisional, config.reference)
    selected = set(selected_ids)
    final_results: list[AdaptiveBatchCaseResult] = []
    for item in provisional:
        if item.error is not None:
            final_results.append(item)
            continue
        if item.case.case_id not in selected:
            final_results.append(item)
            continue
        try:
            final_results.append(_run_reference_and_attribution(item, config))
        except Exception as exc:  # noqa: BLE001
            if config.fail_fast:
                raise
            final_results.append(_replace_result(item, reference_evaluation_status=ReferenceEvaluationStatus.INVALID, error=str(exc)))
    results = tuple(final_results)
    summary = build_adaptive_batch_summary(results)
    groups = build_adaptive_batch_group_summaries(results)
    points = build_accuracy_cost_points(results)
    nondominated = find_nondominated_accuracy_cost_points(points, error_function=lambda point: _default_error(point))
    worst = build_adaptive_batch_worst_cases(results)
    output_dir = Path(config.output_dir)
    artifact_paths: dict[str, str] = {}
    if config.export_csv:
        artifact_paths.update(export_adaptive_batch_csvs(results, groups, points, output_dir))
    if config.export_json:
        path = output_dir / "diagnostics.json"
        export_adaptive_batch_json(
            cases=case_tuple,
            results=results,
            selected_reference_case_ids=selected_ids,
            summary=summary,
            groups=groups,
            points=points,
            nondominated=nondominated,
            worst=worst,
            config=config,
            path=path,
        )
        artifact_paths["diagnostics_json"] = str(path)
    if config.export_markdown:
        path = output_dir / "report.md"
        write_adaptive_batch_markdown_report(
            results=results,
            summary=summary,
            groups=groups,
            points=points,
            nondominated=nondominated,
            worst=worst,
            path=path,
        )
        artifact_paths["report_markdown"] = str(path)
    return AdaptiveBatchResult(
        cases=case_tuple,
        results=results,
        selected_reference_case_ids=selected_ids,
        summary=summary,
        group_summaries=groups,
        accuracy_cost_points=points,
        nondominated_accuracy_cost_points=nondominated,
        worst_cases=worst,
        artifact_paths=dict(sorted(artifact_paths.items())),
        config=config,
        mujoco_version=_mujoco_version(),
        git_commit=_git_commit(),
    )


def build_adaptive_batch_summary(results: Sequence[AdaptiveBatchCaseResult]) -> AdaptiveBatchSummary:
    """Aggregate batch results with explicit improvement denominators."""
    completed = [item for item in results if item.error is None]
    attrs = [item.primary_attribution for item in completed if item.primary_attribution is not None]
    ratios = [item.run_level_comparison.adaptive_step_ratio for item in completed if item.run_level_comparison is not None]
    savings = [item.run_level_comparison.adaptive_step_saving for item in completed if item.run_level_comparison is not None]
    rest_errors = [attr.adaptive_restitution_error for attr in attrs if attr.adaptive_restitution_error is not None]
    pen_errors = [attr.adaptive_penetration_error for attr in attrs if attr.adaptive_penetration_error is not None]
    dur_errors = [attr.adaptive_duration_error for attr in attrs if attr.adaptive_duration_error is not None]
    rest_num, rest_den = _improvement_fraction(attrs, "restitution_outcome")
    pen_num, pen_den = _improvement_fraction(attrs, "penetration_outcome")
    dur_num, dur_den = _improvement_fraction(attrs, "duration_outcome")
    case_num, case_den = _case_improvement_fraction(attrs)
    return AdaptiveBatchSummary(
        total_case_count=len(results),
        completed_case_count=len(completed),
        invalid_case_count=len(results) - len(completed),
        sphere_plane_case_count=sum(item.case.scene_type is AdaptiveBatchSceneType.SPHERE_PLANE for item in results),
        sphere_sphere_case_count=sum(item.case.scene_type is AdaptiveBatchSceneType.SPHERE_SPHERE for item in results),
        primary_matched_case_count=sum(_primary_matched(item) for item in completed),
        primary_unmatched_case_count=sum(not _primary_matched(item) for item in completed),
        reference_checked_case_count=sum(item.reference_evaluation_status is not ReferenceEvaluationStatus.NOT_CHECKED for item in completed),
        reference_not_checked_case_count=sum(item.reference_evaluation_status is ReferenceEvaluationStatus.NOT_CHECKED for item in completed),
        reference_converged_case_count=sum(item.reference_evaluation_status is ReferenceEvaluationStatus.CONVERGED for item in completed),
        reference_unresolved_case_count=sum(item.reference_evaluation_status is ReferenceEvaluationStatus.NOT_CONVERGED for item in completed),
        primary_scope_case_count=sum(attr.scope is AttributionScope.PRIMARY_IMPACT for attr in attrs),
        fallback_scope_case_count=sum(attr.scope is AttributionScope.FALLBACK_RUN_LEVEL for attr in attrs),
        unavailable_scope_case_count=sum(attr.scope is AttributionScope.UNAVAILABLE for attr in attrs),
        primary_case_outcome_counts=dict(sorted(Counter(attr.case_outcome.value for attr in attrs).items())),
        primary_reason_counts=dict(sorted(Counter(attr.primary_reason.value for attr in attrs).items())),
        restitution_outcome_counts=dict(sorted(Counter(attr.restitution_outcome.value for attr in attrs).items())),
        penetration_outcome_counts=dict(sorted(Counter(attr.penetration_outcome.value for attr in attrs).items())),
        duration_outcome_counts=dict(sorted(Counter(attr.duration_outcome.value for attr in attrs).items())),
        primary_restitution_improvement_rate=_safe_rate(rest_num, rest_den),
        primary_penetration_improvement_rate=_safe_rate(pen_num, pen_den),
        primary_duration_improvement_rate=_safe_rate(dur_num, dur_den),
        primary_case_improvement_rate=_safe_rate(case_num, case_den),
        primary_restitution_improvement_numerator=rest_num,
        primary_restitution_improvement_denominator=rest_den,
        primary_penetration_improvement_numerator=pen_num,
        primary_penetration_improvement_denominator=pen_den,
        primary_duration_improvement_numerator=dur_num,
        primary_duration_improvement_denominator=dur_den,
        primary_case_improvement_numerator=case_num,
        primary_case_improvement_denominator=case_den,
        mean_adaptive_step_ratio=_mean(ratios),
        median_adaptive_step_ratio=_median(ratios),
        maximum_adaptive_step_ratio=None if not ratios else max(ratios),
        mean_adaptive_step_saving=_mean(savings),
        median_adaptive_step_saving=_median(savings),
        mean_primary_restitution_error=_mean(rest_errors),
        maximum_primary_restitution_error=None if not rest_errors else max(rest_errors),
        mean_primary_penetration_error=_mean(pen_errors),
        maximum_primary_penetration_error=None if not pen_errors else max(pen_errors),
        mean_primary_duration_error=_mean(dur_errors),
        maximum_primary_duration_error=None if not dur_errors else max(dur_errors),
    )


def build_adaptive_batch_group_summaries(results: Sequence[AdaptiveBatchCaseResult]) -> tuple[AdaptiveBatchGroupSummary, ...]:
    """Build deterministic group summaries."""
    groups: dict[tuple[str, str], list[AdaptiveBatchCaseResult]] = defaultdict(list)
    for item in results:
        case = item.case
        values = {
            "scene_type": case.scene_type.value,
            "macro_timestep": _float_label(case.macro_timestep),
            "solref": _tuple_label(case.contact_params.solref),
            "impact_speed_range": _impact_speed_range(_provisional_impact_speed(item)),
            "sphere_radius": _float_label(case.sphere_a_radius),
            "sphere_mass": _float_label(case.sphere_a_mass),
        }
        for name, value in values.items():
            groups[(name, value)].append(item)
    rows = []
    for (name, value), items in groups.items():
        attrs = [item.primary_attribution for item in items if item.primary_attribution is not None]
        rest_num, rest_den = _improvement_fraction(attrs, "restitution_outcome")
        pen_num, pen_den = _improvement_fraction(attrs, "penetration_outcome")
        dur_num, dur_den = _improvement_fraction(attrs, "duration_outcome")
        savings = [item.run_level_comparison.adaptive_step_saving for item in items if item.run_level_comparison is not None]
        rest_errors = [attr.adaptive_restitution_error for attr in attrs if attr.adaptive_restitution_error is not None]
        pen_errors = [attr.adaptive_penetration_error for attr in attrs if attr.adaptive_penetration_error is not None]
        rows.append(
            AdaptiveBatchGroupSummary(
                group_name=name,
                group_value=value,
                case_count=len(items),
                reference_converged_count=sum(item.reference_evaluation_status is ReferenceEvaluationStatus.CONVERGED for item in items),
                restitution_improvement_rate=_safe_rate(rest_num, rest_den),
                penetration_improvement_rate=_safe_rate(pen_num, pen_den),
                duration_improvement_rate=_safe_rate(dur_num, dur_den),
                mean_step_saving=_mean(savings),
                mean_restitution_error=_mean(rest_errors),
                mean_penetration_error=_mean(pen_errors),
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.group_name, row.group_value)))


def build_accuracy_cost_points(results: Sequence[AdaptiveBatchCaseResult]) -> tuple[AccuracyCostPoint, ...]:
    """Create accuracy/cost points for converged, attributed cases."""
    points = []
    for item in results:
        attr = item.primary_attribution
        comparison = item.run_level_comparison
        if item.reference_evaluation_status is not ReferenceEvaluationStatus.CONVERGED or attr is None or comparison is None:
            continue
        stats = None if item.adaptive_run is None else item.adaptive_run.adaptive_statistics
        points.append(
            AccuracyCostPoint(
                case_id=item.case.case_id,
                scene_type=item.case.scene_type,
                restitution_error=attr.adaptive_restitution_error,
                penetration_error=attr.adaptive_penetration_error,
                duration_error=attr.adaptive_duration_error,
                adaptive_step_ratio=comparison.adaptive_step_ratio,
                adaptive_step_saving=comparison.adaptive_step_saving,
                maximum_substep_count=0 if stats is None else stats.maximum_substep_count,
                substepped_macro_step_ratio=0.0 if stats is None else stats.substepped_macro_step_ratio,
            )
        )
    return tuple(sorted(points, key=lambda point: point.case_id))


def find_nondominated_accuracy_cost_points(
    points: Sequence[AccuracyCostPoint],
    *,
    error_function: Callable[[AccuracyCostPoint], float],
) -> tuple[AccuracyCostPoint, ...]:
    """Return points not dominated by lower-or-equal error and cost."""
    finite = tuple(point for point in points if math.isfinite(error_function(point)))
    nondominated = []
    for point in finite:
        error = error_function(point)
        dominated = False
        for other in finite:
            if other.case_id == point.case_id:
                continue
            other_error = error_function(other)
            if (
                other_error <= error
                and other.adaptive_step_ratio <= point.adaptive_step_ratio
                and (other_error < error or other.adaptive_step_ratio < point.adaptive_step_ratio)
            ):
                dominated = True
                break
        if not dominated:
            nondominated.append(point)
    return tuple(sorted(nondominated, key=lambda point: (error_function(point), point.adaptive_step_ratio, point.case_id)))


def build_adaptive_batch_worst_cases(results: Sequence[AdaptiveBatchCaseResult]) -> AdaptiveBatchWorstCases:
    """Build deterministic worst-case identifiers."""
    attrs = [(item, item.primary_attribution) for item in results if item.primary_attribution is not None]
    return AdaptiveBatchWorstCases(
        maximum_primary_restitution_error_case_id=_worst_attr(attrs, lambda attr: attr.adaptive_restitution_error),
        maximum_primary_penetration_error_case_id=_worst_attr(attrs, lambda attr: attr.adaptive_penetration_error),
        maximum_primary_duration_error_case_id=_worst_attr(attrs, lambda attr: attr.adaptive_duration_error),
        maximum_adaptive_step_ratio_case_id=_worst_result(results, lambda item: None if item.run_level_comparison is None else item.run_level_comparison.adaptive_step_ratio),
        shortest_prediction_lead_case_id=_best_result(results, lambda item: None if item.primary_attribution is None else item.primary_attribution.prediction_lead_macro_steps),
        maximum_substep_limited_case_id=_worst_result(results, lambda item: None if item.primary_attribution is None else item.primary_attribution.maximum_substep_count),
        reference_least_converged_case_id=_least_converged(results),
        primary_episode_unmatched_case_id=_first_case_id(item for item in results if not _primary_matched(item)),
        adaptive_nonphysical_case_id=_first_case_id(item for item in results if item.adaptive_run is not None and item.adaptive_run.validity is BenchmarkValidity.NONPHYSICAL_REBOUND),
    )


def export_adaptive_batch_csvs(
    results: Sequence[AdaptiveBatchCaseResult],
    groups: Sequence[AdaptiveBatchGroupSummary],
    points: Sequence[AccuracyCostPoint],
    output_dir: str | Path,
) -> dict[str, str]:
    """Export all batch CSV artifacts."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "cases_csv": target / "cases.csv",
        "primary_results_csv": target / "primary_results.csv",
        "group_summary_csv": target / "group_summary.csv",
        "accuracy_cost_csv": target / "accuracy_cost.csv",
        "reference_convergence_csv": target / "reference_convergence.csv",
    }
    _write_csv(paths["cases_csv"], [_case_row(item.case) for item in results])
    _write_csv(paths["primary_results_csv"], [_primary_result_row(item) for item in results])
    _write_csv(paths["group_summary_csv"], [_serializable(row) for row in groups])
    _write_csv(paths["accuracy_cost_csv"], [_serializable(point) for point in points])
    _write_csv(paths["reference_convergence_csv"], [_reference_row(item) for item in results])
    return {key: str(value) for key, value in paths.items()}


def export_adaptive_batch_json(
    *,
    cases: Sequence[AdaptiveBatchCase],
    results: Sequence[AdaptiveBatchCaseResult],
    selected_reference_case_ids: Sequence[str],
    summary: AdaptiveBatchSummary,
    groups: Sequence[AdaptiveBatchGroupSummary],
    points: Sequence[AccuracyCostPoint],
    nondominated: Sequence[AccuracyCostPoint],
    worst: AdaptiveBatchWorstCases,
    config: AdaptiveBatchConfig,
    path: str | Path,
) -> None:
    """Export stable machine-readable batch diagnostics."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "adaptive-primary-batch/v1",
        "mujoco_version": _mujoco_version(),
        "git_commit": _git_commit(),
        "batch_config": _serializable(config),
        "selected_reference_case_ids": list(selected_reference_case_ids),
        "case_configs": [_case_row(case) for case in cases],
        "results": [_result_json(item) for item in results],
        "summary": _serializable(summary),
        "group_summaries": [_serializable(row) for row in groups],
        "accuracy_cost_points": [_serializable(point) for point in points],
        "nondominated_accuracy_cost_points": [_serializable(point) for point in nondominated],
        "worst_cases": _serializable(worst),
        "units": {
            "time": "seconds",
            "length": "meters",
            "mass": "kilograms",
            "velocity": "meters/second",
            "step_ratio": "adaptive physics steps / fixed-fine physics steps",
        },
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf8")


def write_adaptive_batch_markdown_report(
    *,
    results: Sequence[AdaptiveBatchCaseResult],
    summary: AdaptiveBatchSummary,
    groups: Sequence[AdaptiveBatchGroupSummary],
    points: Sequence[AccuracyCostPoint],
    nondominated: Sequence[AccuracyCostPoint],
    worst: AdaptiveBatchWorstCases,
    path: str | Path,
) -> None:
    """Write a Markdown batch report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_adaptive_batch_markdown_report(results, summary, groups, points, nondominated, worst), encoding="utf8")


def build_adaptive_batch_markdown_report(
    results: Sequence[AdaptiveBatchCaseResult],
    summary: AdaptiveBatchSummary,
    groups: Sequence[AdaptiveBatchGroupSummary],
    points: Sequence[AccuracyCostPoint],
    nondominated: Sequence[AccuracyCostPoint],
    worst: AdaptiveBatchWorstCases,
) -> str:
    """Build a deterministic Markdown report."""
    lines = [
        "# Adaptive Primary-Impact Batch Report",
        "",
        "## Executive summary",
        "",
        f"- total cases: {summary.total_case_count}",
        f"- completed / invalid: {summary.completed_case_count} / {summary.invalid_case_count}",
        f"- reference checked / converged: {summary.reference_checked_case_count} / {summary.reference_converged_case_count}",
        f"- primary matched / unmatched: {summary.primary_matched_case_count} / {summary.primary_unmatched_case_count}",
        f"- primary restitution improved: {summary.primary_restitution_improvement_numerator} / {summary.primary_restitution_improvement_denominator} eligible cases = {_pct(summary.primary_restitution_improvement_rate)}",
        f"- primary penetration improved: {summary.primary_penetration_improvement_numerator} / {summary.primary_penetration_improvement_denominator} eligible cases = {_pct(summary.primary_penetration_improvement_rate)}",
        f"- primary duration improved: {summary.primary_duration_improvement_numerator} / {summary.primary_duration_improvement_denominator} eligible cases = {_pct(summary.primary_duration_improvement_rate)}",
        f"- mean / median step saving: {_fmt(summary.mean_adaptive_step_saving)} / {_fmt(summary.median_adaptive_step_saving)}",
        "",
        "## Reference coverage",
        "",
        f"- checked: {summary.reference_checked_case_count}",
        f"- not checked: {summary.reference_not_checked_case_count}",
        f"- converged: {summary.reference_converged_case_count}",
        f"- unresolved: {summary.reference_unresolved_case_count}",
        f"- invalid: {sum(item.reference_evaluation_status is ReferenceEvaluationStatus.INVALID for item in results)}",
        "",
        "## Overall primary results",
        "",
        f"- case outcomes: {dict(summary.primary_case_outcome_counts)}",
        f"- restitution outcomes: {dict(summary.restitution_outcome_counts)}",
        f"- penetration outcomes: {dict(summary.penetration_outcome_counts)}",
        f"- duration outcomes: {dict(summary.duration_outcome_counts)}",
        f"- failure reasons: {dict(summary.primary_reason_counts)}",
        "",
        "## Group analysis",
        "",
    ]
    for row in groups:
        if row.group_name in {"scene_type", "macro_timestep", "solref", "impact_speed_range"}:
            lines.append(
                f"- {row.group_name}={row.group_value}: cases={row.case_count}, converged={row.reference_converged_count}, "
                f"restitution={_pct(row.restitution_improvement_rate)}, penetration={_pct(row.penetration_improvement_rate)}, "
                f"duration={_pct(row.duration_improvement_rate)}, mean saving={_fmt(row.mean_step_saving)}"
            )
    lines.extend(["", "## Accuracy-cost analysis", ""])
    for point in sorted(points, key=lambda p: (_default_error(p), p.adaptive_step_ratio, p.case_id))[:5]:
        lines.append(f"- low error / cost candidate {point.case_id}: error={_fmt(_default_error(point))}, saving={_fmt(point.adaptive_step_saving)}")
    for point in sorted(points, key=lambda p: (-(p.restitution_error or 0.0), -p.adaptive_step_saving, p.case_id))[:5]:
        lines.append(f"- high error / high saving candidate {point.case_id}: restitution error={_fmt(point.restitution_error)}, saving={_fmt(point.adaptive_step_saving)}")
    for point in sorted(points, key=lambda p: (_default_error(p), p.adaptive_step_saving, p.case_id))[:5]:
        lines.append(f"- low error / low saving candidate {point.case_id}: error={_fmt(_default_error(point))}, saving={_fmt(point.adaptive_step_saving)}")
    lines.append(f"- Pareto nondominated cases: {[point.case_id for point in nondominated]}")
    lines.extend(["", "## Worst cases", "", *[f"- {key}: {value}" for key, value in _serializable(worst).items()]])
    unresolved = [item.case.case_id for item in results if item.reference_evaluation_status in {ReferenceEvaluationStatus.NOT_CONVERGED, ReferenceEvaluationStatus.INVALID}]
    lines.extend(["", "## Reference-unresolved cases", "", f"- cases: {unresolved}"])
    lines.extend(["", "## Conclusion", "", *_conclusion_lines(summary, groups)])
    return "\n".join(lines) + "\n"


# Internal orchestration helpers


def _run_provisional_case(case: AdaptiveBatchCase, config: AdaptiveBatchConfig) -> AdaptiveBatchCaseResult:
    benchmark_case = _to_benchmark_case(case)
    dataset = run_contact_benchmark((benchmark_case,), validation=config.benchmark_validation, recommendation=config.recommendation)
    by_mode = {result.mode: result for result in dataset.results}
    episodes = {
        mode: segment_contact_episodes(
            collect_contact_episode_samples(benchmark_case, mode=mode, recommendation=config.recommendation),
            config=config.episode_segmentation,
        )
        for mode in BenchmarkMode
    }
    primary = build_primary_impact_comparison(
        case_id=case.case_id,
        reference=episodes[BenchmarkMode.FIXED_FINE],
        coarse=episodes[BenchmarkMode.FIXED_COARSE],
        adaptive=episodes[BenchmarkMode.ADAPTIVE],
        matching=config.episode_matching,
    )
    fine_primary = _first_primary(episodes[BenchmarkMode.FIXED_FINE])
    ref_slice = () if fine_primary is None else (fine_primary,)
    coarse_match = _first_or_none(match_contact_episodes(reference=ref_slice, comparison=episodes[BenchmarkMode.FIXED_COARSE], config=config.episode_matching))
    adaptive_match = _first_or_none(match_contact_episodes(reference=ref_slice, comparison=episodes[BenchmarkMode.ADAPTIVE], config=config.episode_matching))
    trace = run_adaptive_diagnostic_trace(benchmark_case, recommendation=config.recommendation) if config.collect_substep_samples else None
    return AdaptiveBatchCaseResult(
        case=case,
        coarse_run=by_mode[BenchmarkMode.FIXED_COARSE],
        fine_run=by_mode[BenchmarkMode.FIXED_FINE],
        adaptive_run=by_mode[BenchmarkMode.ADAPTIVE],
        coarse_episodes=episodes[BenchmarkMode.FIXED_COARSE],
        fine_episodes=episodes[BenchmarkMode.FIXED_FINE],
        adaptive_episodes=episodes[BenchmarkMode.ADAPTIVE],
        coarse_primary_match=coarse_match,
        adaptive_primary_match=adaptive_match,
        provisional_primary_comparison=primary,
        reference_evaluation_status=ReferenceEvaluationStatus.NOT_CHECKED,
        primary_reference_convergence=None,
        converged_reference_episode=None,
        primary_attribution=None,
        adaptive_trace=trace,
        run_level_comparison=dataset.comparisons[0],
        run_level_reference_convergence=None,
        error=None,
    )


def _run_reference_and_attribution(item: AdaptiveBatchCaseResult, config: AdaptiveBatchConfig) -> AdaptiveBatchCaseResult:
    benchmark_case = _to_benchmark_case(item.case)
    convergence_config = ReferenceConvergenceConfig(refinement_factors=config.reference.refinement_factors)
    primary_convergence = run_episode_reference_convergence(
        benchmark_case,
        segmentation=config.episode_segmentation,
        recommendation=config.recommendation,
        convergence_config=convergence_config,
        matching_config=config.episode_matching,
    )
    run_convergence = run_reference_convergence(benchmark_case, recommendation=config.recommendation, config=convergence_config)
    status = _reference_status(primary_convergence.overall_status)
    reference_episode = (
        _converged_episode(benchmark_case, config, convergence_config)
        if status is ReferenceEvaluationStatus.CONVERGED
        else None
    )
    primary = item.provisional_primary_comparison
    attribution = attribute_primary_impact_failure(
        PrimaryImpactAttributionInput(
            case_id=item.case.case_id,
            candidate_id=primary.candidate_id if primary is not None else f"{item.case.case_id}_candidate",
            coarse_episode=None if primary is None else primary.coarse_episode,
            adaptive_episode=None if primary is None else primary.adaptive_episode,
            reference_episode=reference_episode if reference_episode is not None else (None if primary is None else primary.reference_episode),
            coarse_match=item.coarse_primary_match,
            adaptive_match=item.adaptive_primary_match,
            primary_comparison=primary,
            primary_reference_convergence=primary_convergence,
            run_level_comparison=item.run_level_comparison,
            run_level_reference_convergence=run_convergence,
            adaptive_trace=item.adaptive_trace,
        ),
        improvement_config=config.improvement,
        attribution_config=config.attribution,
    )
    return _replace_result(
        item,
        reference_evaluation_status=status,
        primary_reference_convergence=primary_convergence,
        converged_reference_episode=reference_episode,
        primary_attribution=attribution,
        run_level_reference_convergence=run_convergence,
    )


def _select_reference_case_ids(results: Sequence[AdaptiveBatchCaseResult], config: BatchReferenceConfig) -> tuple[str, ...]:
    valid = [item for item in results if item.error is None]
    if config.mode is ReferenceEvaluationMode.NONE:
        return ()
    if config.mode is ReferenceEvaluationMode.ALL:
        return tuple(sorted(item.case.case_id for item in valid))
    selected: dict[str, None] = {}

    def add(items: Iterable[AdaptiveBatchCaseResult]) -> None:
        for item in items:
            if config.maximum_selected_cases is not None and len(selected) >= config.maximum_selected_cases:
                return
            selected.setdefault(item.case.case_id, None)

    if config.include_all_nonphysical_coarse:
        add(sorted((item for item in valid if item.coarse_run is not None and item.coarse_run.validity is BenchmarkValidity.NONPHYSICAL_REBOUND), key=lambda item: item.case.case_id))
    if config.include_all_primary_not_improved:
        add(sorted((item for item in valid if _provisional_not_improved(item)), key=lambda item: item.case.case_id))
    add(_top_errors(valid, lambda item: None if item.provisional_primary_comparison is None else item.provisional_primary_comparison.adaptive_restitution_error, config.top_k_adaptive_restitution_error))
    add(_top_errors(valid, lambda item: None if item.provisional_primary_comparison is None else item.provisional_primary_comparison.adaptive_penetration_error, config.top_k_adaptive_penetration_error))
    add(_top_errors(valid, lambda item: None if item.provisional_primary_comparison is None else item.provisional_primary_comparison.adaptive_duration_error, config.top_k_adaptive_duration_error))
    add(sorted((item for item in valid if item.case.scene_type is AdaptiveBatchSceneType.SPHERE_SPHERE), key=lambda item: item.case.case_id))
    add(_boundary_cases(valid))
    return tuple(selected)


def _to_benchmark_case(case: AdaptiveBatchCase):
    if case.scene_type is AdaptiveBatchSceneType.SPHERE_PLANE:
        plane_z = 0.0 if case.plane is None else case.plane.point[2]
        height = case.sphere_a_initial_position[2] - plane_z
        return SpherePlaneBenchmarkCase(
            case_id=case.case_id,
            initial_height=height,
            macro_timestep=case.macro_timestep,
            solref=case.contact_params.solref,
            radius=case.sphere_a_radius,
            total_simulation_time=case.total_simulation_time,
            solimp=case.contact_params.solimp,
        )
    speed = abs(case.sphere_a_initial_velocity[0])
    separation = abs(case.sphere_b_initial_position[0] - case.sphere_a_initial_position[0]) if case.sphere_b_initial_position else 0.6
    return SphereSphereBenchmarkCase(
        case_id=case.case_id,
        macro_timestep=case.macro_timestep,
        solref=case.contact_params.solref,
        radius=case.sphere_a_radius,
        initial_separation=separation,
        speed=speed,
        total_simulation_time=case.total_simulation_time,
        solimp=case.contact_params.solimp,
    )


# Serialization and small helpers


def _sphere_plane_case(height: float, timestep: float, solref: tuple[float, float], radius: float, mass: float, *, adaptive_config: AdaptiveSubstepConfig) -> AdaptiveBatchCase:
    return AdaptiveBatchCase(
        case_id=f"sp_h{height:g}_dt{_dt_label(timestep)}_solref{solref[0]:g}_{solref[1]:g}_r{radius:g}_m{mass:g}",
        scene_type=AdaptiveBatchSceneType.SPHERE_PLANE,
        macro_timestep=timestep,
        total_simulation_time=0.9 if height <= 0.7 else 1.1,
        contact_params=MuJoCoContactSolverParams(solref=solref, solimp=(0.9, 0.9, 0.001, 0.5, 2.0)),
        adaptive_config=adaptive_config,
        sphere_a_radius=radius,
        sphere_a_mass=mass,
        sphere_a_initial_position=(0.0, 0.0, height),
        sphere_a_initial_velocity=(0.0, 0.0, 0.0),
        plane=AnalyticPlane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)),
        metadata={"height": height, "drop_height": height},
    )


def _sphere_sphere_case(label: str, mass_a: float, mass_b: float, vel_a: float, vel_b: float, radius: float, solref: tuple[float, float], *, adaptive_config: AdaptiveSubstepConfig) -> AdaptiveBatchCase:
    return AdaptiveBatchCase(
        case_id=f"ss_{label}_dt240_solref{solref[0]:g}_{solref[1]:g}",
        scene_type=AdaptiveBatchSceneType.SPHERE_SPHERE,
        macro_timestep=1.0 / 240.0,
        total_simulation_time=0.6,
        contact_params=MuJoCoContactSolverParams(solref=solref, solimp=(0.9, 0.9, 0.001, 0.5, 2.0)),
        adaptive_config=adaptive_config,
        sphere_a_radius=radius,
        sphere_a_mass=mass_a,
        sphere_a_initial_position=(-0.3, 0.0, 0.0),
        sphere_a_initial_velocity=(vel_a, 0.0, 0.0),
        sphere_b_radius=radius,
        sphere_b_mass=mass_b,
        sphere_b_initial_position=(0.3, 0.0, 0.0),
        sphere_b_initial_velocity=(vel_b, 0.0, 0.0),
        metadata={"relative_speed": abs(vel_a - vel_b)},
    )


def _deterministic_layered_sample(combos: Sequence[tuple[float, float, tuple[float, float], float, float]], config: BatchGenerationConfig, *, target_count: int):
    ordered = sorted(combos, key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    required = []
    seen = set()
    dimensions = [
        lambda item: item[0],
        lambda item: item[1],
        lambda item: item[2],
        lambda item: item[3],
        lambda item: item[4],
    ]
    for dimension in dimensions:
        for value in sorted({dimension(item) for item in ordered}, key=repr):
            for item in ordered:
                if dimension(item) == value and item not in seen:
                    required.append(item)
                    seen.add(item)
                    break
    offset = config.sampling_seed % len(ordered) if ordered else 0
    rotated = ordered[offset:] + ordered[:offset]
    for item in rotated:
        if len(required) >= target_count:
            break
        if item not in seen:
            required.append(item)
            seen.add(item)
    return required


def _validate_unique_cases(cases: Sequence[AdaptiveBatchCase]) -> tuple[AdaptiveBatchCase, ...]:
    result = tuple(cases)
    ids = [case.case_id for case in result]
    duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise PhysicsValidationError(f"case_id must be unique; duplicates={duplicates}")
    return result


def _invalid_case_result(case: AdaptiveBatchCase, error: str) -> AdaptiveBatchCaseResult:
    return AdaptiveBatchCaseResult(case, None, None, None, (), (), (), None, None, None, ReferenceEvaluationStatus.INVALID, None, None, None, None, None, None, error)


def _replace_result(item: AdaptiveBatchCaseResult, **changes) -> AdaptiveBatchCaseResult:
    data = item.__dict__.copy()
    data.update(changes)
    return AdaptiveBatchCaseResult(**data)


def _reference_status(status: ReferenceConvergenceStatus) -> ReferenceEvaluationStatus:
    if status is ReferenceConvergenceStatus.CONVERGED:
        return ReferenceEvaluationStatus.CONVERGED
    if status is ReferenceConvergenceStatus.NOT_CONVERGED:
        return ReferenceEvaluationStatus.NOT_CONVERGED
    return ReferenceEvaluationStatus.INVALID


def _converged_episode(benchmark_case, config: AdaptiveBatchConfig, convergence_config: ReferenceConvergenceConfig) -> ContactEpisodeMetrics | None:
    factor = max(convergence_config.refinement_factors)
    recommendation = SubstepRecommendationConfig(maximum_substeps=config.recommendation.maximum_substeps * factor)
    episodes = segment_contact_episodes(
        collect_contact_episode_samples(benchmark_case, mode=BenchmarkMode.FIXED_FINE, recommendation=recommendation),
        config=config.episode_segmentation,
    )
    return _first_primary(episodes)


def _first_primary(episodes: Sequence[ContactEpisodeMetrics]) -> ContactEpisodeMetrics | None:
    for episode in episodes:
        if episode.kind.value == "primary_impact":
            return episode
    return episodes[0] if episodes else None


def _first_or_none(items: Sequence[EpisodeMatch]) -> EpisodeMatch | None:
    return items[0] if items else None


def _primary_matched(item: AdaptiveBatchCaseResult) -> bool:
    return item.adaptive_primary_match is not None and item.adaptive_primary_match.status is EpisodeMatchStatus.MATCHED


def _provisional_not_improved(item: AdaptiveBatchCaseResult) -> bool:
    primary = item.provisional_primary_comparison
    if primary is None:
        return False
    return primary.adaptive_improves_restitution is False or primary.adaptive_improves_penetration is False


def _top_errors(items: Sequence[AdaptiveBatchCaseResult], getter: Callable[[AdaptiveBatchCaseResult], float | None], k: int) -> tuple[AdaptiveBatchCaseResult, ...]:
    scored = [(getter(item), item) for item in items]
    scored = [(score, item) for score, item in scored if score is not None and math.isfinite(score)]
    scored.sort(key=lambda pair: (-pair[0], pair[1].case.case_id))
    return tuple(item for _, item in scored[:k])


def _boundary_cases(items: Sequence[AdaptiveBatchCaseResult]) -> tuple[AdaptiveBatchCaseResult, ...]:
    rows = []
    for getter in (
        lambda item: item.case.macro_timestep,
        lambda item: item.case.sphere_a_radius,
        lambda item: item.case.sphere_a_mass,
    ):
        ordered = sorted(items, key=lambda item: (getter(item), item.case.case_id))
        if ordered:
            rows.append(ordered[0])
            rows.append(ordered[-1])
    return tuple(rows)


def _improvement_fraction(attrs: Sequence[PrimaryImpactFailureAttribution], field_name: str) -> tuple[int, int]:
    eligible = [
        getattr(attr, field_name)
        for attr in attrs
        if attr.primary_reference_status is ReferenceConvergenceStatus.CONVERGED
        and attr.adaptive_primary_match_status is EpisodeMatchStatus.MATCHED
        and getattr(attr, field_name) is not ImprovementOutcome.NOT_APPLICABLE
        and getattr(attr, field_name) is not ImprovementOutcome.REFERENCE_UNRESOLVED
        and getattr(attr, field_name) is not ImprovementOutcome.EPISODE_UNMATCHED
    ]
    return sum(value is ImprovementOutcome.IMPROVED or value is ImprovementOutcome.BOTH_ACCEPTABLE for value in eligible), len(eligible)


def _case_improvement_fraction(attrs: Sequence[PrimaryImpactFailureAttribution]) -> tuple[int, int]:
    eligible = [
        attr.case_outcome
        for attr in attrs
        if attr.primary_reference_status is ReferenceConvergenceStatus.CONVERGED
        and attr.adaptive_primary_match_status is EpisodeMatchStatus.MATCHED
        and attr.case_outcome
        not in {
            PrimaryImpactCaseOutcome.REFERENCE_UNRESOLVED,
            PrimaryImpactCaseOutcome.EPISODE_UNMATCHED,
            PrimaryImpactCaseOutcome.INVALID_ADAPTIVE,
        }
    ]
    return sum(value in {PrimaryImpactCaseOutcome.IMPROVED, PrimaryImpactCaseOutcome.BOTH_ACCEPTABLE} for value in eligible), len(eligible)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _default_error(point: AccuracyCostPoint) -> float:
    values = [value for value in (point.restitution_error, point.penetration_error, point.duration_error) if value is not None]
    return math.inf if not values else sum(values)


def _worst_attr(pairs, getter):
    scored = [(getter(attr), item.case.case_id) for item, attr in pairs]
    scored = [(score, case_id) for score, case_id in scored if score is not None and math.isfinite(score)]
    return None if not scored else sorted(scored, key=lambda pair: (-pair[0], pair[1]))[0][1]


def _worst_result(results, getter):
    scored = [(getter(item), item.case.case_id) for item in results]
    scored = [(score, case_id) for score, case_id in scored if score is not None and math.isfinite(float(score))]
    return None if not scored else sorted(scored, key=lambda pair: (-float(pair[0]), pair[1]))[0][1]


def _best_result(results, getter):
    scored = [(getter(item), item.case.case_id) for item in results]
    scored = [(score, case_id) for score, case_id in scored if score is not None and math.isfinite(float(score))]
    return None if not scored else sorted(scored, key=lambda pair: (float(pair[0]), pair[1]))[0][1]


def _least_converged(results):
    unresolved = [item.case.case_id for item in results if item.reference_evaluation_status in {ReferenceEvaluationStatus.NOT_CONVERGED, ReferenceEvaluationStatus.INVALID}]
    return sorted(unresolved)[0] if unresolved else None


def _first_case_id(items: Iterable[AdaptiveBatchCaseResult]) -> str | None:
    ids = sorted(item.case.case_id for item in items)
    return ids[0] if ids else None


def _provisional_impact_speed(item: AdaptiveBatchCaseResult) -> float | None:
    if item.provisional_primary_comparison is not None:
        return item.provisional_primary_comparison.reference_episode.impact_speed
    if item.fine_run is not None:
        return item.fine_run.impact_speed
    return None


def _impact_speed_range(speed: float | None) -> str:
    if speed is None:
        return "unknown"
    if speed < 2.0:
        return "low"
    if speed < 4.0:
        return "medium"
    return "high"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _case_row(case: AdaptiveBatchCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "scene_type": case.scene_type.value,
        "macro_timestep": case.macro_timestep,
        "total_simulation_time": case.total_simulation_time,
        "solref": _tuple_label(case.contact_params.solref),
        "solimp": _tuple_label(case.contact_params.solimp),
        "sphere_a_radius": case.sphere_a_radius,
        "sphere_a_mass": case.sphere_a_mass,
        "sphere_a_initial_position": _tuple_label(case.sphere_a_initial_position),
        "sphere_a_initial_velocity": _tuple_label(case.sphere_a_initial_velocity),
        "sphere_b_radius": case.sphere_b_radius,
        "sphere_b_mass": case.sphere_b_mass,
        "sphere_b_initial_position": None if case.sphere_b_initial_position is None else _tuple_label(case.sphere_b_initial_position),
        "sphere_b_initial_velocity": None if case.sphere_b_initial_velocity is None else _tuple_label(case.sphere_b_initial_velocity),
        "adaptive_config": json.dumps(_serializable(case.adaptive_config), sort_keys=True),
        "metadata": json.dumps(dict(case.metadata), sort_keys=True),
    }


def _primary_result_row(item: AdaptiveBatchCaseResult) -> dict[str, object]:
    attr = item.primary_attribution
    comparison = item.run_level_comparison
    trace = item.adaptive_trace
    return {
        "case_id": item.case.case_id,
        "reference_status": item.reference_evaluation_status.value,
        "scope": None if attr is None else attr.scope.value,
        "case_outcome": None if attr is None else attr.case_outcome.value,
        "primary_reason": None if attr is None else attr.primary_reason.value,
        "restitution_outcome": None if attr is None else attr.restitution_outcome.value,
        "penetration_outcome": None if attr is None else attr.penetration_outcome.value,
        "duration_outcome": None if attr is None else attr.duration_outcome.value,
        "adaptive_restitution_error": None if attr is None else attr.adaptive_restitution_error,
        "adaptive_penetration_error": None if attr is None else attr.adaptive_penetration_error,
        "adaptive_duration_error": None if attr is None else attr.adaptive_duration_error,
        "adaptive_step_ratio": None if comparison is None else comparison.adaptive_step_ratio,
        "adaptive_step_saving": None if comparison is None else comparison.adaptive_step_saving,
        "prediction_lead_macro_steps": None if attr is None else attr.prediction_lead_macro_steps,
        "maximum_substep_count": None if trace is None else trace.global_maximum_substep_count,
        "coarse_episode_count": len(item.coarse_episodes),
        "fine_episode_count": len(item.fine_episodes),
        "adaptive_episode_count": len(item.adaptive_episodes),
        "error": item.error,
    }


def _reference_row(item: AdaptiveBatchCaseResult) -> dict[str, object]:
    conv = item.primary_reference_convergence
    return {
        "case_id": item.case.case_id,
        "reference_status": item.reference_evaluation_status.value,
        "level_count": None if conv is None else len(conv.levels),
        "overall_status": None if conv is None else conv.overall_status.value,
        "restitution_status": None if conv is None else conv.restitution.status.value,
        "penetration_status": None if conv is None else conv.maximum_penetration.status.value,
        "duration_status": None if conv is None else conv.contact_duration.status.value,
    }


def _result_json(item: AdaptiveBatchCaseResult) -> dict[str, object]:
    return {
        "case": _case_row(item.case),
        "coarse_run": _serializable(item.coarse_run),
        "fine_run": _serializable(item.fine_run),
        "adaptive_run": _serializable(item.adaptive_run),
        "coarse_episodes": [_serializable(ep) for ep in item.coarse_episodes],
        "fine_episodes": [_serializable(ep) for ep in item.fine_episodes],
        "adaptive_episodes": [_serializable(ep) for ep in item.adaptive_episodes],
        "coarse_primary_match": _serializable(item.coarse_primary_match),
        "adaptive_primary_match": _serializable(item.adaptive_primary_match),
        "provisional_primary_comparison": _serializable(item.provisional_primary_comparison),
        "reference_evaluation_status": item.reference_evaluation_status.value,
        "primary_reference_convergence": _serializable(item.primary_reference_convergence),
        "converged_reference_episode": _serializable(item.converged_reference_episode),
        "primary_attribution": _serializable(item.primary_attribution),
        "adaptive_trace": _serializable(item.adaptive_trace),
        "run_level_comparison": _serializable(item.run_level_comparison),
        "run_level_reference_convergence": _serializable(item.run_level_reference_convergence),
        "error": item.error,
    }


def _serializable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _serializable(val) for key, val in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _serializable(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    return str(value)


def _vector3(value: Sequence[float] | None, field_name: str) -> Vector3:
    if value is None or len(value) != 3:
        raise PhysicsValidationError(f"{field_name} must be a 3D vector")
    return tuple(_finite_float(component, field_name=f"{field_name}[{index}]") for index, component in enumerate(value))  # type: ignore[return-value]


def _dt_label(timestep: float) -> str:
    return str(round(1.0 / timestep))


def _float_label(value: float) -> str:
    return f"{value:.8g}"


def _tuple_label(values: Sequence[float]) -> str:
    return ",".join(_float_label(value) for value in values)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


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


def _conclusion_lines(summary: AdaptiveBatchSummary, groups: Sequence[AdaptiveBatchGroupSummary]) -> list[str]:
    lines = []
    stable = [row for row in groups if row.group_name == "solref" and row.restitution_improvement_rate is not None]
    if stable:
        best = sorted(stable, key=lambda row: (-(row.restitution_improvement_rate or 0.0), row.group_value))[0]
        lines.append(f"- most stable solref group in this run: {best.group_value} with restitution improvement {_pct(best.restitution_improvement_rate)}.")
    if summary.reference_unresolved_case_count:
        lines.append("- unresolved reference cases remain and should be diagnosed before treating them as adaptive failures.")
    if summary.primary_case_improvement_denominator and summary.primary_case_improvement_rate is not None:
        lines.append(f"- primary-impact improvement was measured on {summary.primary_case_improvement_denominator} eligible cases.")
    lines.append("- extending new geometry is best postponed until reference coverage and primary matching are stable for this batch.")
    return lines
