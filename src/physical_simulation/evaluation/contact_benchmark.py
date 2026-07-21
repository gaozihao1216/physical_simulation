"""Benchmark adaptive MuJoCo substepping against fixed-step baselines."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

from physical_simulation.assets import Transform, create_ground, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation.contact_calibration import RestitutionMeasurement, RestitutionOutcome
from physical_simulation.mujoco import (
    AdaptiveMuJoCoRunner,
    AdaptiveSubstepConfig,
    AnalyticPlane,
    ContactMotionState,
    MuJoCoContactSolverParams,
    SpherePlaneAdaptiveCandidate,
    SphereSphereAdaptiveCandidate,
    SubstepRecommendationConfig,
)
from physical_simulation.scene import AssetInstanceSpec, PhysicsSceneSpec, create_scene
from physical_simulation.runtime import SimulationStepResult
from physical_simulation.validation.asset_validator import _finite_float
from physical_simulation.validation.errors import PhysicsValidationError

Vector3 = tuple[float, float, float]


class BenchmarkMode(Enum):
    """Execution mode for contact benchmark runs."""

    FIXED_COARSE = "fixed_coarse"
    FIXED_FINE = "fixed_fine"
    ADAPTIVE = "adaptive"


class BenchmarkValidity(Enum):
    """Diagnostic validity class for one benchmark run."""

    VALID = "valid"
    NONPHYSICAL_REBOUND = "nonphysical_rebound"
    EXCESSIVE_PENETRATION = "excessive_penetration"
    TIMEOUT = "timeout"
    UNSTABLE = "unstable"


@dataclass(frozen=True)
class BenchmarkValidationConfig:
    """Thresholds used to classify benchmark validity."""

    maximum_restitution: float = 1.05
    maximum_normalized_penetration: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum_restitution",
            _finite_float(
                self.maximum_restitution,
                field_name="maximum_restitution",
                minimum=0.0,
                strict_minimum=True,
                error_type=PhysicsValidationError,
            ),
        )
        object.__setattr__(
            self,
            "maximum_normalized_penetration",
            _finite_float(
                self.maximum_normalized_penetration,
                field_name="maximum_normalized_penetration",
                minimum=0.0,
                strict_minimum=True,
                error_type=PhysicsValidationError,
            ),
        )


@dataclass(frozen=True)
class AdaptiveRunStatistics:
    """Statistics collected from an adaptive benchmark run."""

    state_macro_steps: dict[str, int]
    substep_count_distribution: dict[int, int]
    maximum_substep_count: int
    first_approaching_time: float | None
    first_contact_time: float | None
    prediction_lead_time: float | None
    substepped_macro_step_ratio: float


@dataclass(frozen=True)
class ContactBenchmarkResult:
    """Result for one mode in one contact benchmark case."""

    case_id: str
    mode: BenchmarkMode
    validity: BenchmarkValidity
    timestep: float
    macro_timestep: float
    total_simulation_time: float
    outcome: RestitutionOutcome
    impact_speed: float | None
    rebound_speed: float | None
    restitution: float | None
    maximum_penetration: float
    normalized_penetration: float | None
    contact_duration_seconds: float | None
    final_position: Vector3
    final_linear_velocity: Vector3
    final_angular_velocity: Vector3
    macro_step_count: int
    physics_step_count: int
    wall_time_seconds: float
    adaptive_substepped_macro_steps: int | None
    adaptive_max_substep_count: int | None
    normal_energy_ratio: float | None = None
    adaptive_statistics: AdaptiveRunStatistics | None = None


@dataclass(frozen=True)
class BenchmarkComparison:
    """Compare fixed coarse and adaptive runs against fixed fine."""

    case_id: str
    coarse_restitution_error: float | None
    adaptive_restitution_error: float | None
    coarse_penetration_error: float
    adaptive_penetration_error: float
    coarse_rebound_velocity_error: float | None
    adaptive_rebound_velocity_error: float | None
    adaptive_step_ratio: float
    adaptive_step_saving: float
    adaptive_improves_restitution: bool | None
    adaptive_improves_penetration: bool


@dataclass(frozen=True)
class SpherePlaneBenchmarkCase:
    """Sphere dropping onto a static analytic plane."""

    case_id: str
    initial_height: float
    macro_timestep: float
    solref: tuple[float, float]
    radius: float = 0.1
    total_simulation_time: float = 1.0
    solimp: tuple[float, float, float, float, float] = (0.9, 0.9, 0.001, 0.5, 2.0)


@dataclass(frozen=True)
class SphereSphereBenchmarkCase:
    """Two dynamic spheres moving head-on toward each other."""

    case_id: str
    macro_timestep: float
    solref: tuple[float, float]
    radius: float = 0.1
    initial_separation: float = 0.6
    speed: float = 1.5
    total_simulation_time: float = 0.6
    solimp: tuple[float, float, float, float, float] = (0.9, 0.9, 0.001, 0.5, 2.0)


ContactBenchmarkCase = SpherePlaneBenchmarkCase | SphereSphereBenchmarkCase


@dataclass(frozen=True)
class ContactBenchmarkDataset:
    """Complete benchmark dataset including raw results and comparisons."""

    config: dict[str, object]
    mujoco_version: str
    cases: tuple[dict[str, object], ...]
    results: tuple[ContactBenchmarkResult, ...]
    comparisons: tuple[BenchmarkComparison, ...]
    units: dict[str, str]


def generate_default_benchmark_cases(
    *,
    include_regression_case: bool = True,
) -> tuple[ContactBenchmarkCase, ...]:
    """Create the standard Phase 2G4 benchmark case set."""
    cases: list[ContactBenchmarkCase] = []
    for height in (0.4, 0.7, 1.0, 1.3):
        for macro_timestep in (1.0 / 120.0, 1.0 / 240.0, 1.0 / 480.0):
            for solref in ((0.02, 0.3), (0.02, 0.5), (0.01, 0.3)):
                cases.append(
                    SpherePlaneBenchmarkCase(
                        case_id=f"sphere_plane_h{height:g}_dt{_dt_label(macro_timestep)}_solref{solref[0]:g}_{solref[1]:g}",
                        initial_height=height,
                        macro_timestep=macro_timestep,
                        solref=solref,
                    )
                )
    if include_regression_case:
        cases.append(
            SpherePlaneBenchmarkCase(
                case_id="sphere_plane_regression_h1_dt240_solref0.005_0.3",
                initial_height=1.0,
                macro_timestep=1.0 / 240.0,
                solref=(0.005, 0.3),
            )
        )
    cases.append(
        SphereSphereBenchmarkCase(
            case_id="sphere_sphere_headon_dt240_solref0.01_0.3",
            macro_timestep=1.0 / 240.0,
            solref=(0.01, 0.3),
        )
    )
    return tuple(cases)


def run_contact_benchmark(
    cases: Iterable[ContactBenchmarkCase],
    *,
    validation: BenchmarkValidationConfig = BenchmarkValidationConfig(),
    recommendation: SubstepRecommendationConfig = SubstepRecommendationConfig(maximum_substeps=16),
) -> ContactBenchmarkDataset:
    """Run fixed coarse, fixed fine, and adaptive benchmark modes for all cases."""
    case_tuple = tuple(cases)
    results: list[ContactBenchmarkResult] = []
    comparisons: list[BenchmarkComparison] = []
    for case in case_tuple:
        case_results = tuple(
            _run_case_mode(case, mode, validation=validation, recommendation=recommendation)
            for mode in BenchmarkMode
        )
        results.extend(case_results)
        comparisons.append(compare_benchmark_results(case_results))
    return ContactBenchmarkDataset(
        config={
            "validation": asdict(validation),
            "recommendation": asdict(recommendation),
            "modes": [mode.value for mode in BenchmarkMode],
        },
        mujoco_version=_mujoco_version(),
        cases=tuple(_case_to_dict(case) for case in case_tuple),
        results=tuple(results),
        comparisons=tuple(comparisons),
        units=_units(),
    )


def classify_benchmark_validity(
    *,
    outcome: RestitutionOutcome,
    restitution: float | None,
    normalized_penetration: float | None,
    validation: BenchmarkValidationConfig = BenchmarkValidationConfig(),
    unstable: bool = False,
) -> BenchmarkValidity:
    """Classify one run without treating e > 1 as a valid material response."""
    if unstable or _bad_optional(restitution) or _bad_optional(normalized_penetration):
        return BenchmarkValidity.UNSTABLE
    if outcome is RestitutionOutcome.TIMEOUT:
        return BenchmarkValidity.TIMEOUT
    if restitution is not None and restitution > validation.maximum_restitution:
        return BenchmarkValidity.NONPHYSICAL_REBOUND
    if (
        normalized_penetration is not None
        and normalized_penetration > validation.maximum_normalized_penetration
    ):
        return BenchmarkValidity.EXCESSIVE_PENETRATION
    return BenchmarkValidity.VALID


def compare_benchmark_results(results: Sequence[ContactBenchmarkResult]) -> BenchmarkComparison:
    """Compare a three-mode benchmark result set against fixed fine."""
    by_mode = {result.mode: result for result in results}
    coarse = by_mode[BenchmarkMode.FIXED_COARSE]
    fine = by_mode[BenchmarkMode.FIXED_FINE]
    adaptive = by_mode[BenchmarkMode.ADAPTIVE]
    coarse_restitution_error = _optional_abs_error(coarse.restitution, fine.restitution)
    adaptive_restitution_error = _optional_abs_error(adaptive.restitution, fine.restitution)
    coarse_rebound_error = _optional_abs_error(coarse.rebound_speed, fine.rebound_speed)
    adaptive_rebound_error = _optional_abs_error(adaptive.rebound_speed, fine.rebound_speed)
    adaptive_step_ratio = adaptive.physics_step_count / fine.physics_step_count
    adaptive_step_saving = 1.0 - adaptive_step_ratio
    adaptive_improves_restitution = (
        None
        if coarse_restitution_error is None or adaptive_restitution_error is None
        else adaptive_restitution_error <= coarse_restitution_error
    )
    return BenchmarkComparison(
        case_id=fine.case_id,
        coarse_restitution_error=coarse_restitution_error,
        adaptive_restitution_error=adaptive_restitution_error,
        coarse_penetration_error=abs(coarse.maximum_penetration - fine.maximum_penetration),
        adaptive_penetration_error=abs(adaptive.maximum_penetration - fine.maximum_penetration),
        coarse_rebound_velocity_error=coarse_rebound_error,
        adaptive_rebound_velocity_error=adaptive_rebound_error,
        adaptive_step_ratio=adaptive_step_ratio,
        adaptive_step_saving=adaptive_step_saving,
        adaptive_improves_restitution=adaptive_improves_restitution,
        adaptive_improves_penetration=abs(adaptive.maximum_penetration - fine.maximum_penetration)
        <= abs(coarse.maximum_penetration - fine.maximum_penetration),
    )


def export_benchmark_csv(results: Sequence[ContactBenchmarkResult], path: str | Path) -> None:
    """Export one CSV row per benchmark run."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [_result_to_row(result) for result in results]
    fieldnames = list(rows[0].keys()) if rows else list(_result_to_row(_empty_result()).keys())
    with target.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_benchmark_json(dataset: ContactBenchmarkDataset, path: str | Path) -> None:
    """Export the complete benchmark dataset as stable JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_dataset_to_dict(dataset), indent=2, sort_keys=True), encoding="utf8")


def write_benchmark_markdown_report(dataset: ContactBenchmarkDataset, path: str | Path) -> None:
    """Write a compact Markdown report with summaries and worst cases."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_benchmark_markdown_report(dataset), encoding="utf8")


def build_benchmark_markdown_report(dataset: ContactBenchmarkDataset) -> str:
    """Build a Markdown benchmark report string."""
    counts = Counter(result.validity.value for result in dataset.results)
    comparisons = dataset.comparisons
    adaptive_restitution_errors = [c.adaptive_restitution_error for c in comparisons if c.adaptive_restitution_error is not None]
    adaptive_penetration_errors = [c.adaptive_penetration_error for c in comparisons]
    step_ratios = [c.adaptive_step_ratio for c in comparisons]
    coarse_failures = [
        result for result in dataset.results
        if result.mode is BenchmarkMode.FIXED_COARSE and result.validity is BenchmarkValidity.NONPHYSICAL_REBOUND
    ]
    penetration_failures = [
        result for result in dataset.results
        if result.validity is BenchmarkValidity.EXCESSIVE_PENETRATION
    ]
    not_improved = [
        comparison for comparison in comparisons
        if comparison.adaptive_improves_restitution is False or not comparison.adaptive_improves_penetration
    ]
    worst_coarse = _max_by(comparisons, lambda c: c.coarse_restitution_error)
    worst_adaptive = _max_by(comparisons, lambda c: c.adaptive_restitution_error)
    lines = [
        "# Adaptive Contact Benchmark",
        "",
        f"- total cases: {len(dataset.cases)}",
        f"- total runs: {len(dataset.results)}",
        f"- validity counts: {dict(sorted(counts.items()))}",
        f"- adaptive restitution error mean/max: {_fmt(_mean(adaptive_restitution_errors))} / {_fmt(_max(adaptive_restitution_errors))}",
        f"- adaptive penetration error mean/max: {_fmt(_mean(adaptive_penetration_errors))} / {_fmt(_max(adaptive_penetration_errors))}",
        f"- adaptive step ratio mean: {_fmt(_mean(step_ratios))}",
        f"- adaptive step saving mean: {_fmt(1.0 - _mean(step_ratios) if step_ratios else None)}",
        "",
        "## Per-Case Comparison",
        "",
        "| case | coarse e error | adaptive e error | coarse penetration error | adaptive penetration error | adaptive step ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for comparison in comparisons:
        lines.append(
            f"| {comparison.case_id} | {_fmt(comparison.coarse_restitution_error)} | "
            f"{_fmt(comparison.adaptive_restitution_error)} | {_fmt(comparison.coarse_penetration_error)} | "
            f"{_fmt(comparison.adaptive_penetration_error)} | {_fmt(comparison.adaptive_step_ratio)} |"
        )
    lines.extend([
        "",
        "## Diagnostics",
        "",
        f"- e > 1 cases: {', '.join(result.case_id for result in coarse_failures) or 'none'}",
        f"- excessive penetration cases: {', '.join(result.case_id for result in penetration_failures) or 'none'}",
        f"- adaptive not improved cases: {', '.join(item.case_id for item in not_improved) or 'none'}",
        f"- worst coarse restitution case: {worst_coarse.case_id if worst_coarse else 'none'}",
        f"- worst adaptive restitution case: {worst_adaptive.case_id if worst_adaptive else 'none'}",
    ])
    return "\n".join(lines) + "\n"


def improvement_rates(comparisons: Sequence[BenchmarkComparison]) -> dict[str, float]:
    """Return aggregate improvement rates over comparable cases."""
    restitution_flags = [item.adaptive_improves_restitution for item in comparisons if item.adaptive_improves_restitution is not None]
    penetration_flags = [item.adaptive_improves_penetration for item in comparisons]
    return {
        "restitution": sum(bool(value) for value in restitution_flags) / len(restitution_flags) if restitution_flags else 0.0,
        "penetration": sum(bool(value) for value in penetration_flags) / len(penetration_flags) if penetration_flags else 0.0,
    }


def _run_case_mode(
    case: ContactBenchmarkCase,
    mode: BenchmarkMode,
    *,
    validation: BenchmarkValidationConfig,
    recommendation: SubstepRecommendationConfig,
) -> ContactBenchmarkResult:
    params = _params(case)
    macro_steps = _macro_steps(case)
    fine_substeps = _fixed_fine_substeps(recommendation)
    timestep = case.macro_timestep if mode is not BenchmarkMode.FIXED_FINE else case.macro_timestep / fine_substeps
    backend = MuJoCoBackend()
    start = time.perf_counter()
    try:
        backend.load_scene(_scene_for_case(case, timestep=timestep))
        _apply_initial_velocity(case, backend, update_initial=True)
        if mode is BenchmarkMode.ADAPTIVE:
            result, measurement, physics_steps, stats = _run_adaptive(case, backend, recommendation, macro_steps)
            adaptive_substeps = sum(count for substeps, count in stats.substep_count_distribution.items() if substeps > 1)
            adaptive_max = stats.maximum_substep_count
        else:
            steps = macro_steps if mode is BenchmarkMode.FIXED_COARSE else macro_steps * fine_substeps
            samples = _run_fixed(case, backend, steps)
            result = samples[-1]
            measurement = _measure_samples(case, samples)
            physics_steps = steps
            stats = None
            adaptive_substeps = None
            adaptive_max = None
        wall_time = time.perf_counter() - start
        state = result.get_body_state(_primary_body_id(case))
        validity = classify_benchmark_validity(
            outcome=measurement.outcome,
            restitution=measurement.measured_restitution,
            normalized_penetration=measurement.normalized_penetration,
            validation=validation,
        )
        return ContactBenchmarkResult(
            case_id=case.case_id,
            mode=mode,
            validity=validity,
            timestep=timestep,
            macro_timestep=case.macro_timestep,
            total_simulation_time=macro_steps * case.macro_timestep,
            outcome=measurement.outcome,
            impact_speed=measurement.impact_speed,
            rebound_speed=measurement.rebound_speed,
            restitution=measurement.measured_restitution,
            maximum_penetration=measurement.maximum_penetration_depth,
            normalized_penetration=measurement.normalized_penetration,
            contact_duration_seconds=measurement.contact_duration_seconds,
            final_position=state.position,
            final_linear_velocity=state.linear_velocity,
            final_angular_velocity=state.angular_velocity,
            macro_step_count=macro_steps,
            physics_step_count=physics_steps,
            wall_time_seconds=wall_time,
            adaptive_substepped_macro_steps=adaptive_substeps,
            adaptive_max_substep_count=adaptive_max,
            normal_energy_ratio=None if measurement.measured_restitution is None else measurement.measured_restitution ** 2,
            adaptive_statistics=stats,
        )
    except Exception:
        wall_time = time.perf_counter() - start
        return _unstable_result(case, mode, timestep, macro_steps, wall_time)
    finally:
        backend.close()


def _run_fixed(
    case: ContactBenchmarkCase,
    backend: MuJoCoBackend,
    steps: int,
) -> list[SimulationStepResult]:
    samples = [backend.reset()]
    for _ in range(steps):
        samples.append(backend.step())
    return samples


def _run_adaptive(
    case: ContactBenchmarkCase,
    backend: MuJoCoBackend,
    recommendation: SubstepRecommendationConfig,
    macro_steps: int,
) -> tuple[SimulationStepResult, RestitutionMeasurement, int, AdaptiveRunStatistics]:
    runner = AdaptiveMuJoCoRunner(
        backend,
        candidates=(_candidate_for_case(case),),
        config=AdaptiveSubstepConfig(
            macro_timestep=case.macro_timestep,
            recommendation=recommendation,
            resting_window_macro_steps=3,
            separating_hold_macro_steps=1,
        ),
    )
    samples = [runner.reset()]
    state_counts: Counter[str] = Counter()
    substep_counts: Counter[int] = Counter()
    first_approaching_time: float | None = None
    first_contact_time: float | None = None
    first_prediction_lead: float | None = None
    result = samples[-1]
    for _ in range(macro_steps):
        adaptive = runner.step()
        decision = adaptive.decision
        result = adaptive.advance_result.simulation_result
        state_counts[decision.state_after.value] += 1
        substep_counts[decision.substep_count] += 1
        if first_approaching_time is None and decision.state_after is ContactMotionState.APPROACHING:
            first_approaching_time = result.time
            if decision.prediction is not None:
                first_prediction_lead = decision.prediction.time_to_contact
        for sample in adaptive.substep_results:
            if first_contact_time is None and _contacts_for_case(case, sample):
                first_contact_time = sample.time
        samples.extend(adaptive.substep_results or (result,))
    measurement = _measure_samples(case, samples)
    substepped = sum(count for substeps, count in substep_counts.items() if substeps > 1)
    stats = AdaptiveRunStatistics(
        state_macro_steps=dict(sorted(state_counts.items())),
        substep_count_distribution=dict(sorted(substep_counts.items())),
        maximum_substep_count=max(substep_counts) if substep_counts else 1,
        first_approaching_time=first_approaching_time,
        first_contact_time=first_contact_time,
        prediction_lead_time=None if first_prediction_lead is None else first_prediction_lead,
        substepped_macro_step_ratio=substepped / macro_steps if macro_steps else 0.0,
    )
    return result, measurement, runner.physics_step_count, stats


def _measure_samples(case: ContactBenchmarkCase, samples: Sequence[SimulationStepResult]) -> RestitutionMeasurement:
    last_approach_speed = 0.0
    impact_speed = 0.0
    contact_start: int | None = None
    contact_start_time: float | None = None
    contact_end: int | None = None
    contact_end_time: float | None = None
    last_contact: int | None = None
    last_contact_time: float | None = None
    max_penetration = 0.0
    observed = 0
    for sample in samples:
        normal_speed = _normal_speed(case, sample)
        contacts = _contacts_for_case(case, sample)
        if contact_start is None and normal_speed < 0.0:
            last_approach_speed = -normal_speed
        if contacts:
            if contact_start is None:
                contact_start = sample.step_index
                contact_start_time = sample.time
                impact_speed = last_approach_speed
            last_contact = sample.step_index
            last_contact_time = sample.time
            observed += 1
            max_penetration = max(max_penetration, max(contact.penetration_depth for contact in contacts))
        elif contact_start is not None and contact_end is None and last_contact is not None:
            contact_end = last_contact
            contact_end_time = last_contact_time
        if contact_end is not None and normal_speed > 1.0e-6 and not contacts:
            rebound = normal_speed
            duration_steps = contact_end - contact_start + 1
            return RestitutionMeasurement(
                runtime_body_id=_primary_body_id(case),
                outcome=RestitutionOutcome.REBOUNDED,
                impact_speed=impact_speed,
                rebound_speed=rebound,
                measured_restitution=rebound / impact_speed if impact_speed > 0.0 else None,
                contact_start_step=contact_start,
                contact_end_step=contact_end,
                contact_duration_steps=duration_steps,
                contact_duration_seconds=None if contact_start_time is None or contact_end_time is None else contact_end_time - contact_start_time,
                maximum_penetration_depth=max_penetration,
                normalized_penetration=max_penetration / _characteristic_length(case),
                observed_contact_steps=observed,
            )
    return RestitutionMeasurement(
        runtime_body_id=_primary_body_id(case),
        outcome=RestitutionOutcome.TIMEOUT,
        impact_speed=impact_speed,
        rebound_speed=None,
        measured_restitution=None,
        contact_start_step=-1 if contact_start is None else contact_start,
        contact_end_step=contact_end,
        contact_duration_steps=None,
        contact_duration_seconds=None,
        maximum_penetration_depth=max_penetration,
        normalized_penetration=max_penetration / _characteristic_length(case),
        observed_contact_steps=observed,
    )


def _scene_for_case(case: ContactBenchmarkCase, *, timestep: float) -> PhysicsSceneSpec:
    params = _params(case)
    if isinstance(case, SpherePlaneBenchmarkCase):
        ground = create_single_body_asset(
            asset_id="ground_asset",
            body=_body_with_params(create_ground("ground"), params),
        )
        sphere = create_single_body_asset(
            asset_id="sphere_asset",
            body=_body_with_params(create_sphere("sphere", case.radius, mass=1.0, create_visual=False), params),
        )
        return create_scene(
            scene_id=case.case_id,
            instances=(
                AssetInstanceSpec("ground_01", ground, fixed_base=True),
                AssetInstanceSpec("sphere_01", sphere, Transform(position=(0.0, 0.0, case.initial_height))),
            ),
            timestep=timestep,
        )
    sphere_a = create_single_body_asset(
        asset_id="sphere_a_asset",
        body=_body_with_params(create_sphere("sphere_a", case.radius, mass=1.0, create_visual=False), params),
    )
    sphere_b = create_single_body_asset(
        asset_id="sphere_b_asset",
        body=_body_with_params(create_sphere("sphere_b", case.radius, mass=1.0, create_visual=False), params),
    )
    half = case.initial_separation / 2.0
    return create_scene(
        scene_id=case.case_id,
        instances=(
            AssetInstanceSpec("sphere_a_01", sphere_a, Transform(position=(-half, 0.0, 0.0))),
            AssetInstanceSpec("sphere_b_01", sphere_b, Transform(position=(half, 0.0, 0.0))),
        ),
        gravity=(0.0, 0.0, 0.0),
        timestep=timestep,
    )


def _candidate_for_case(case: ContactBenchmarkCase):
    params = _params(case)
    if isinstance(case, SpherePlaneBenchmarkCase):
        return SpherePlaneAdaptiveCandidate(
            f"{case.case_id}_candidate",
            "sphere_01/sphere",
            case.radius,
            AnalyticPlane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)),
            params,
        )
    return SphereSphereAdaptiveCandidate(
        f"{case.case_id}_candidate",
        "sphere_a_01/sphere_a",
        case.radius,
        "sphere_b_01/sphere_b",
        case.radius,
        params,
    )


def _apply_initial_velocity(case: ContactBenchmarkCase, backend: MuJoCoBackend, *, update_initial: bool) -> None:
    if isinstance(case, SphereSphereBenchmarkCase):
        backend.set_body_velocity("sphere_a_01/sphere_a", (case.speed, 0.0, 0.0), update_initial=update_initial)
        backend.set_body_velocity("sphere_b_01/sphere_b", (-case.speed, 0.0, 0.0), update_initial=update_initial)


def _contacts_for_case(case: ContactBenchmarkCase, sample: SimulationStepResult):
    if isinstance(case, SpherePlaneBenchmarkCase):
        body_id = "sphere_01/sphere"
        return tuple(contact for contact in sample.contacts if contact.body_a == body_id or contact.body_b == body_id)
    return tuple(
        contact
        for contact in sample.contacts
        if {contact.body_a, contact.body_b} == {"sphere_a_01/sphere_a", "sphere_b_01/sphere_b"}
    )


def _normal_speed(case: ContactBenchmarkCase, sample: SimulationStepResult) -> float:
    if isinstance(case, SpherePlaneBenchmarkCase):
        return sample.get_body_state("sphere_01/sphere").linear_velocity[2]
    first = sample.get_body_state("sphere_a_01/sphere_a")
    second = sample.get_body_state("sphere_b_01/sphere_b")
    offset = tuple(second.position[index] - first.position[index] for index in range(3))
    normal = _normalize(offset)
    relative = tuple(second.linear_velocity[index] - first.linear_velocity[index] for index in range(3))
    return _dot(relative, normal)


def _body_with_params(body, params: MuJoCoContactSolverParams):
    from dataclasses import replace

    return replace(body, colliders=tuple(replace(collider, mujoco_contact_params=params) for collider in body.colliders))


def _params(case: ContactBenchmarkCase) -> MuJoCoContactSolverParams:
    return MuJoCoContactSolverParams(solref=case.solref, solimp=case.solimp)


def _macro_steps(case: ContactBenchmarkCase) -> int:
    return max(1, round(case.total_simulation_time / case.macro_timestep))


def _fixed_fine_substeps(config: SubstepRecommendationConfig) -> int:
    """Use the configured finest adaptive grid as the fixed-fine baseline."""
    return config.maximum_substeps


def _primary_body_id(case: ContactBenchmarkCase) -> str:
    return "sphere_01/sphere" if isinstance(case, SpherePlaneBenchmarkCase) else "sphere_a_01/sphere_a"


def _characteristic_length(case: ContactBenchmarkCase) -> float:
    return case.radius


def _case_to_dict(case: ContactBenchmarkCase) -> dict[str, object]:
    data = asdict(case)
    data["case_type"] = "sphere_plane" if isinstance(case, SpherePlaneBenchmarkCase) else "sphere_sphere"
    return data


def _dataset_to_dict(dataset: ContactBenchmarkDataset) -> dict[str, object]:
    return {
        "config": dataset.config,
        "mujoco_version": dataset.mujoco_version,
        "cases": list(dataset.cases),
        "results": [_result_to_json(result) for result in dataset.results],
        "comparisons": [_comparison_to_json(comparison) for comparison in dataset.comparisons],
        "units": dataset.units,
        "field_semantics": {
            "normal_energy_ratio": "eta_E = e^2, a sphere-plane normal-collision diagnostic only",
            "wall_time_seconds": "measured CPU wall time; not a strict regression metric",
            "physics_step_count": "number of MuJoCo mj_step calls and primary cost metric",
        },
    }


def _result_to_json(result: ContactBenchmarkResult) -> dict[str, object]:
    data = _result_to_row(result)
    if result.adaptive_statistics is not None:
        data["adaptive_statistics"] = asdict(result.adaptive_statistics)
    return data


def _result_to_row(result: ContactBenchmarkResult) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "mode": result.mode.value,
        "validity": result.validity.value,
        "timestep": result.timestep,
        "macro_timestep": result.macro_timestep,
        "total_simulation_time": result.total_simulation_time,
        "outcome": result.outcome.value,
        "impact_speed": result.impact_speed,
        "rebound_speed": result.rebound_speed,
        "restitution": result.restitution,
        "normal_energy_ratio": result.normal_energy_ratio,
        "maximum_penetration": result.maximum_penetration,
        "normalized_penetration": result.normalized_penetration,
        "contact_duration_seconds": result.contact_duration_seconds,
        "final_position": list(result.final_position),
        "final_linear_velocity": list(result.final_linear_velocity),
        "final_angular_velocity": list(result.final_angular_velocity),
        "macro_step_count": result.macro_step_count,
        "physics_step_count": result.physics_step_count,
        "wall_time_seconds": result.wall_time_seconds,
        "adaptive_substepped_macro_steps": result.adaptive_substepped_macro_steps,
        "adaptive_max_substep_count": result.adaptive_max_substep_count,
    }


def _comparison_to_json(comparison: BenchmarkComparison) -> dict[str, object]:
    return asdict(comparison)


def _empty_result() -> ContactBenchmarkResult:
    return ContactBenchmarkResult(
        case_id="",
        mode=BenchmarkMode.FIXED_COARSE,
        validity=BenchmarkValidity.UNSTABLE,
        timestep=0.0,
        macro_timestep=0.0,
        total_simulation_time=0.0,
        outcome=RestitutionOutcome.TIMEOUT,
        impact_speed=None,
        rebound_speed=None,
        restitution=None,
        maximum_penetration=0.0,
        normalized_penetration=None,
        contact_duration_seconds=None,
        final_position=(math.nan, math.nan, math.nan),
        final_linear_velocity=(math.nan, math.nan, math.nan),
        final_angular_velocity=(math.nan, math.nan, math.nan),
        macro_step_count=0,
        physics_step_count=0,
        wall_time_seconds=0.0,
        adaptive_substepped_macro_steps=None,
        adaptive_max_substep_count=None,
    )


def _unstable_result(
    case: ContactBenchmarkCase,
    mode: BenchmarkMode,
    timestep: float,
    macro_steps: int,
    wall_time: float,
) -> ContactBenchmarkResult:
    return ContactBenchmarkResult(
        case_id=case.case_id,
        mode=mode,
        validity=BenchmarkValidity.UNSTABLE,
        timestep=timestep,
        macro_timestep=case.macro_timestep,
        total_simulation_time=macro_steps * case.macro_timestep,
        outcome=RestitutionOutcome.TIMEOUT,
        impact_speed=None,
        rebound_speed=None,
        restitution=None,
        maximum_penetration=math.nan,
        normalized_penetration=None,
        contact_duration_seconds=None,
        final_position=(math.nan, math.nan, math.nan),
        final_linear_velocity=(math.nan, math.nan, math.nan),
        final_angular_velocity=(math.nan, math.nan, math.nan),
        macro_step_count=macro_steps,
        physics_step_count=0,
        wall_time_seconds=wall_time,
        adaptive_substepped_macro_steps=None,
        adaptive_max_substep_count=None,
    )


def _units() -> dict[str, str]:
    return {
        "timestep": "s",
        "macro_timestep": "s",
        "total_simulation_time": "s",
        "impact_speed": "m/s",
        "rebound_speed": "m/s",
        "maximum_penetration": "m",
        "contact_duration_seconds": "s",
        "final_position": "m",
        "final_linear_velocity": "m/s",
        "final_angular_velocity": "rad/s",
        "wall_time_seconds": "s",
    }


def _mujoco_version() -> str:
    try:
        import mujoco

        return str(getattr(mujoco, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def _dt_label(value: float) -> str:
    reciprocal = round(1.0 / value)
    return f"1over{reciprocal}"


def _optional_abs_error(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return abs(first - second)


def _bad_optional(value: float | None) -> bool:
    return value is not None and not math.isfinite(value)


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(first[index] * second[index] for index in range(3))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector3) -> Vector3:
    length = _norm(vector)
    if length <= 1.0e-12:
        return (1.0, 0.0, 0.0)
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.6g}"


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _max(values: Sequence[float]) -> float | None:
    return None if not values else max(values)


def _max_by(values: Sequence[BenchmarkComparison], selector) -> BenchmarkComparison | None:
    comparable = [(value, selector(value)) for value in values if selector(value) is not None]
    if not comparable:
        return None
    return max(comparable, key=lambda item: item[1])[0]
