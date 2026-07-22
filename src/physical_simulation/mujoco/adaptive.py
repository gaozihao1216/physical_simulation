"""Adaptive MuJoCo substepping based on explicit analytic collision candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from physical_simulation.backends.errors import BackendNotLoadedError, MuJoCoRuntimeError
from physical_simulation.backends.mujoco_backend import MuJoCoBackend
from physical_simulation.mujoco.collision_prediction import (
    AnalyticPlane,
    CollisionPrediction,
    SolverCollisionEstimate,
    Vector3,
    estimate_solver_collision,
    predict_sphere_plane_collision,
    predict_sphere_sphere_collision,
)
from physical_simulation.mujoco.contact_params import MuJoCoContactSolverParams
from physical_simulation.mujoco.contact_timescale import (
    SubstepRecommendation,
    SubstepRecommendationConfig,
    estimate_solver_contact_timescale,
    recommend_solver_substeps,
)
from physical_simulation.mujoco.substepping import MuJoCoSubstepRunner, SubstepAdvanceResult
from physical_simulation.runtime import SimulationStepResult
from physical_simulation.validation.asset_validator import _finite_float, _non_empty_string
from physical_simulation.validation.errors import PhysicsValidationError


class ContactMotionState(Enum):
    """Coarse contact motion state used by AdaptiveMuJoCoRunner."""

    FREE = "free"
    APPROACHING = "approaching"
    IMPACTING = "impacting"
    RESTING = "resting"
    SEPARATING = "separating"


@dataclass(frozen=True)
class FinitePlaneBounds:
    """Finite rectangular bounds for an analytic plane approximation."""

    center: Vector3
    axis_u: Vector3
    axis_v: Vector3
    half_extent_u: float
    half_extent_v: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vector3(self.center, "center"))
        object.__setattr__(self, "axis_u", _normalize(_vector3(self.axis_u, "axis_u")))
        object.__setattr__(self, "axis_v", _normalize(_vector3(self.axis_v, "axis_v")))
        object.__setattr__(
            self,
            "half_extent_u",
            _finite_float(self.half_extent_u, field_name="half_extent_u", minimum=0.0, strict_minimum=True, error_type=PhysicsValidationError),
        )
        object.__setattr__(
            self,
            "half_extent_v",
            _finite_float(self.half_extent_v, field_name="half_extent_v", minimum=0.0, strict_minimum=True, error_type=PhysicsValidationError),
        )


@dataclass(frozen=True)
class SpherePlaneAdaptiveCandidate:
    """Explicit sphere-plane candidate for adaptive substep decisions."""

    candidate_id: str
    sphere_runtime_body_id: str
    sphere_radius: float
    plane: AnalyticPlane
    contact_params: MuJoCoContactSolverParams
    finite_bounds: FinitePlaneBounds | None = None

    def __post_init__(self) -> None:
        _validate_candidate_id(self.candidate_id)
        _validate_body_id(self.sphere_runtime_body_id, "sphere_runtime_body_id")
        object.__setattr__(
            self,
            "sphere_radius",
            _finite_float(
                self.sphere_radius,
                field_name="sphere_radius",
                minimum=0.0,
                strict_minimum=True,
                error_type=PhysicsValidationError,
            ),
        )
        if not isinstance(self.plane, AnalyticPlane):
            raise PhysicsValidationError(f"plane must be AnalyticPlane; actual value={self.plane!r}")
        if self.finite_bounds is not None and not isinstance(self.finite_bounds, FinitePlaneBounds):
            raise PhysicsValidationError(f"finite_bounds must be FinitePlaneBounds or None; actual value={self.finite_bounds!r}")
        _validate_contact_params(self.contact_params)


@dataclass(frozen=True)
class SphereSphereAdaptiveCandidate:
    """Explicit sphere-sphere candidate for adaptive substep decisions."""

    candidate_id: str
    body_a_id: str
    radius_a: float
    body_b_id: str
    radius_b: float
    contact_params: MuJoCoContactSolverParams

    def __post_init__(self) -> None:
        _validate_candidate_id(self.candidate_id)
        _validate_body_id(self.body_a_id, "body_a_id")
        _validate_body_id(self.body_b_id, "body_b_id")
        if self.body_a_id == self.body_b_id:
            raise PhysicsValidationError("body_a_id and body_b_id must be different")
        object.__setattr__(
            self,
            "radius_a",
            _finite_float(
                self.radius_a,
                field_name="radius_a",
                minimum=0.0,
                strict_minimum=True,
                error_type=PhysicsValidationError,
            ),
        )
        object.__setattr__(
            self,
            "radius_b",
            _finite_float(
                self.radius_b,
                field_name="radius_b",
                minimum=0.0,
                strict_minimum=True,
                error_type=PhysicsValidationError,
            ),
        )
        _validate_contact_params(self.contact_params)


@dataclass(frozen=True)
class ConservativePrimitiveAdaptiveCandidate:
    """Conservative body-pair candidate based on world-space bounding spheres."""

    candidate_id: str
    body_a_id: str
    bounding_radius_a: float
    body_b_id: str
    bounding_radius_b: float
    contact_params: MuJoCoContactSolverParams

    def __post_init__(self) -> None:
        _validate_candidate_id(self.candidate_id)
        _validate_body_id(self.body_a_id, "body_a_id")
        _validate_body_id(self.body_b_id, "body_b_id")
        if self.body_a_id == self.body_b_id:
            raise PhysicsValidationError("body_a_id and body_b_id must be different")
        object.__setattr__(
            self,
            "bounding_radius_a",
            _finite_float(self.bounding_radius_a, field_name="bounding_radius_a", minimum=0.0, strict_minimum=True, error_type=PhysicsValidationError),
        )
        object.__setattr__(
            self,
            "bounding_radius_b",
            _finite_float(self.bounding_radius_b, field_name="bounding_radius_b", minimum=0.0, strict_minimum=True, error_type=PhysicsValidationError),
        )
        _validate_contact_params(self.contact_params)


AdaptiveCollisionCandidate = SpherePlaneAdaptiveCandidate | SphereSphereAdaptiveCandidate | ConservativePrimitiveAdaptiveCandidate


@dataclass(frozen=True)
class AdaptiveSubstepConfig:
    """Configuration for explicit-candidate adaptive substepping."""

    macro_timestep: float = 1.0 / 240.0
    prediction_horizon_multiplier: float = 1.5
    minimum_approach_speed: float = 0.05
    resting_normal_speed_threshold: float = 0.01
    resting_linear_speed_threshold: float = 0.01
    resting_angular_speed_threshold: float = 0.01
    resting_window_macro_steps: int = 3
    separating_hold_macro_steps: int = 1
    recommendation: SubstepRecommendationConfig = SubstepRecommendationConfig()

    def __post_init__(self) -> None:
        for field_name, minimum, strict in (
            ("macro_timestep", 0.0, True),
            ("prediction_horizon_multiplier", 0.0, True),
            ("minimum_approach_speed", 0.0, False),
            ("resting_normal_speed_threshold", 0.0, False),
            ("resting_linear_speed_threshold", 0.0, False),
            ("resting_angular_speed_threshold", 0.0, False),
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_float(
                    getattr(self, field_name),
                    field_name=field_name,
                    minimum=minimum,
                    strict_minimum=strict,
                    error_type=PhysicsValidationError,
                ),
            )
        for field_name in ("resting_window_macro_steps", "separating_hold_macro_steps"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PhysicsValidationError(f"{field_name} must be a non-negative int; actual value={value!r}")
        if self.resting_window_macro_steps < 1:
            raise PhysicsValidationError(
                f"resting_window_macro_steps must be >= 1; actual value={self.resting_window_macro_steps!r}"
            )
        if not isinstance(self.recommendation, SubstepRecommendationConfig):
            raise PhysicsValidationError(
                f"recommendation must be SubstepRecommendationConfig; actual value={self.recommendation!r}"
            )


@dataclass(frozen=True)
class AdaptiveStepDecision:
    """Explain one adaptive macro-step substep decision."""

    state_before: ContactMotionState
    state_after: ContactMotionState
    selected_candidate_id: str | None
    prediction: CollisionPrediction | None
    solver_estimate: SolverCollisionEstimate | None
    substep_count: int
    macro_timestep: float
    actual_substep_timestep: float
    active_contact_observed: bool
    reason: str


@dataclass(frozen=True)
class AdaptiveStepResult:
    """Result of one adaptive macro step."""

    decision: AdaptiveStepDecision
    advance_result: SubstepAdvanceResult
    substep_results: tuple[SimulationStepResult, ...] = ()


class AdaptiveMuJoCoRunner:
    """Adaptive macro-step runner for explicitly registered simple collision candidates."""

    @classmethod
    def from_scene(
        cls,
        backend: MuJoCoBackend,
        *,
        scene,
        runner_config: AdaptiveSubstepConfig,
        candidate_build_config=None,
        manual_candidates: Sequence[AdaptiveCollisionCandidate] = (),
    ) -> "AdaptiveMuJoCoRunner":
        """Create a runner from automatic scene candidates plus optional manual candidates."""
        from physical_simulation.mujoco.adaptive_candidates import create_adaptive_runner_from_scene

        runner, _result = create_adaptive_runner_from_scene(
            backend,
            scene=scene,
            runner_config=runner_config,
            candidate_build_config=candidate_build_config,
            manual_candidates=manual_candidates,
        )
        return runner

    def __init__(
        self,
        backend: MuJoCoBackend,
        *,
        candidates: Sequence[AdaptiveCollisionCandidate],
        config: AdaptiveSubstepConfig,
    ) -> None:
        if not isinstance(backend, MuJoCoBackend):
            raise MuJoCoRuntimeError(f"backend must be MuJoCoBackend; actual value={backend!r}")
        if not backend.is_loaded:
            raise BackendNotLoadedError("AdaptiveMuJoCoRunner requires a loaded MuJoCoBackend")
        if not isinstance(config, AdaptiveSubstepConfig):
            raise PhysicsValidationError(f"config must be AdaptiveSubstepConfig; actual value={config!r}")
        self._backend = backend
        self._config = config
        self._substep_runner = MuJoCoSubstepRunner(backend, macro_timestep=config.macro_timestep)
        self._candidates = tuple(candidates)
        self._candidate_by_id: dict[str, AdaptiveCollisionCandidate] = {}
        for candidate in self._candidates:
            if not isinstance(candidate, (SpherePlaneAdaptiveCandidate, SphereSphereAdaptiveCandidate, ConservativePrimitiveAdaptiveCandidate)):
                raise PhysicsValidationError(f"unsupported adaptive candidate; actual value={candidate!r}")
            if candidate.candidate_id in self._candidate_by_id:
                raise PhysicsValidationError(f"candidate_id values must be unique; duplicate={candidate.candidate_id!r}")
            self._candidate_by_id[candidate.candidate_id] = candidate
        self._state = ContactMotionState.FREE
        self._active_candidate_id: str | None = None
        self._cached_recommendations: dict[str, SubstepRecommendation] = {}
        self._resting_counts: dict[str, int] = {}
        self._separating_remaining = 0

    @property
    def state(self) -> ContactMotionState:
        """Return the current contact motion state."""
        return self._state

    @property
    def macro_step_index(self) -> int:
        """Return completed macro steps."""
        return self._substep_runner.macro_step_index

    @property
    def physics_step_count(self) -> int:
        """Return cumulative internal MuJoCo physics steps."""
        return self._substep_runner.physics_step_count

    def reset(self) -> SimulationStepResult:
        """Reset backend, runner counters, and adaptive state."""
        result = self._substep_runner.reset()
        self._state = ContactMotionState.FREE
        self._active_candidate_id = None
        self._cached_recommendations.clear()
        self._resting_counts.clear()
        self._separating_remaining = 0
        return result

    def step(self) -> AdaptiveStepResult:
        """Advance one macro step using adaptive substep selection."""
        current = self._backend._build_step_result()
        state_before = self._state
        horizon = self._config.macro_timestep * self._config.prediction_horizon_multiplier
        pre_active = self._active_candidates(current)
        estimates = self._candidate_estimates(current, horizon)
        selected = self._select_estimate(estimates)
        selected_candidate_id = selected.prediction_candidate_id if selected is not None else None
        solver_estimate = selected.estimate if selected is not None else None
        prediction = solver_estimate.prediction if solver_estimate is not None else None

        recommendation = solver_estimate.recommendation if solver_estimate is not None else None
        if selected_candidate_id is not None and recommendation is not None:
            self._cached_recommendations[selected_candidate_id] = recommendation
            self._active_candidate_id = selected_candidate_id

        if recommendation is None:
            recommendation = self._state_recommendation(state_before, pre_active)

        substep_count = recommendation.substep_count if recommendation is not None else 1
        reason = self._decision_reason(state_before, selected_candidate_id, pre_active, recommendation)
        samples: list[SimulationStepResult] = []
        advance = self._substep_runner.step(substep_count=substep_count, substep_callback=samples.append)
        observed = tuple(sample for sample in samples if self._candidate_contacts(sample))
        final_result = advance.simulation_result
        final_active = self._active_candidates(final_result)
        active_observed = bool(observed) or bool(final_active)
        state_after = self._update_state(
            state_before=state_before,
            selected_candidate_id=selected_candidate_id,
            pre_active=pre_active,
            final_active=final_active,
            observed_active=active_observed,
            prediction=prediction,
            final_result=final_result,
        )
        self._state = state_after
        decision = AdaptiveStepDecision(
            state_before=state_before,
            state_after=state_after,
            selected_candidate_id=self._active_candidate_id if selected_candidate_id is None else selected_candidate_id,
            prediction=prediction,
            solver_estimate=solver_estimate,
            substep_count=substep_count,
            macro_timestep=self._config.macro_timestep,
            actual_substep_timestep=self._config.macro_timestep / substep_count,
            active_contact_observed=active_observed,
            reason=reason,
        )
        return AdaptiveStepResult(decision=decision, advance_result=advance, substep_results=tuple(samples))

    def _candidate_estimates(self, result: SimulationStepResult, horizon: float) -> tuple["_CandidateEstimate", ...]:
        estimates: list[_CandidateEstimate] = []
        for candidate in self._candidates:
            prediction = self._predict_candidate(candidate, result, horizon)
            if prediction is None or prediction.normal_approach_speed < self._config.minimum_approach_speed:
                continue
            estimate = estimate_solver_collision(
                prediction=prediction,
                params=candidate.contact_params,
                macro_timestep=self._config.macro_timestep,
                config=self._config.recommendation,
            )
            estimates.append(_CandidateEstimate(candidate.candidate_id, estimate))
        return tuple(estimates)

    def _select_estimate(self, estimates: tuple["_CandidateEstimate", ...]) -> "_CandidateEstimate | None":
        if not estimates:
            return None
        return min(
            estimates,
            key=lambda item: (item.estimate.recommendation.actual_substep_timestep, item.prediction_candidate_id),
        )

    def _predict_candidate(
        self,
        candidate: AdaptiveCollisionCandidate,
        result: SimulationStepResult,
        horizon: float,
    ) -> CollisionPrediction | None:
        if isinstance(candidate, SpherePlaneAdaptiveCandidate):
            state = result.get_body_state(candidate.sphere_runtime_body_id)
            prediction = predict_sphere_plane_collision(
                sphere_position=state.position,
                sphere_velocity=state.linear_velocity,
                sphere_radius=candidate.sphere_radius,
                plane=candidate.plane,
                prediction_horizon=horizon,
            )
            if prediction is None or candidate.finite_bounds is None:
                return prediction
            contact_point = _subtract(
                _add(state.position, _scale(state.linear_velocity, prediction.time_to_contact)),
                _scale(candidate.plane.normal, candidate.sphere_radius),
            )
            return prediction if _point_in_bounds(contact_point, candidate.finite_bounds) else None
        if isinstance(candidate, ConservativePrimitiveAdaptiveCandidate):
            first = result.get_body_state(candidate.body_a_id)
            second = result.get_body_state(candidate.body_b_id)
            return _predict_conservative_bounding_spheres(
                body_a_position=first.position,
                body_a_velocity=first.linear_velocity,
                body_a_angular_velocity=first.angular_velocity,
                radius_a=candidate.bounding_radius_a,
                body_b_position=second.position,
                body_b_velocity=second.linear_velocity,
                body_b_angular_velocity=second.angular_velocity,
                radius_b=candidate.bounding_radius_b,
                prediction_horizon=horizon,
            )
        first = result.get_body_state(candidate.body_a_id)
        second = result.get_body_state(candidate.body_b_id)
        return predict_sphere_sphere_collision(
            sphere_a_position=first.position,
            sphere_a_velocity=first.linear_velocity,
            sphere_a_radius=candidate.radius_a,
            sphere_b_position=second.position,
            sphere_b_velocity=second.linear_velocity,
            sphere_b_radius=candidate.radius_b,
            prediction_horizon=horizon,
        )

    def _state_recommendation(
        self,
        state: ContactMotionState,
        pre_active: tuple[str, ...],
    ) -> SubstepRecommendation | None:
        if state is ContactMotionState.RESTING:
            return None
        candidate_id = self._active_candidate_id
        if pre_active:
            candidate_id = pre_active[0]
            self._active_candidate_id = candidate_id
        if state in {ContactMotionState.APPROACHING, ContactMotionState.IMPACTING, ContactMotionState.SEPARATING}:
            if candidate_id in self._cached_recommendations:
                return self._cached_recommendations[candidate_id]
        if pre_active and candidate_id in self._candidate_by_id:
            recommendation = self._recommend_from_params(self._candidate_by_id[candidate_id].contact_params)
            self._cached_recommendations[candidate_id] = recommendation
            return recommendation
        return None

    def _recommend_from_params(self, params: MuJoCoContactSolverParams) -> SubstepRecommendation:
        timescale = estimate_solver_contact_timescale(params)
        return recommend_solver_substeps(
            macro_timestep=self._config.macro_timestep,
            timescale=timescale,
            params=params,
            config=self._config.recommendation,
        )

    def _active_candidates(self, result: SimulationStepResult) -> tuple[str, ...]:
        active: list[str] = []
        contacts = result.contacts
        for candidate in self._candidates:
            if any(self._contact_matches_candidate(contact.body_a, contact.body_b, candidate) for contact in contacts):
                active.append(candidate.candidate_id)
        return tuple(active)

    def _candidate_contacts(self, result: SimulationStepResult) -> tuple[str, ...]:
        return self._active_candidates(result)

    def _contact_matches_candidate(
        self,
        first_body: str,
        second_body: str,
        candidate: AdaptiveCollisionCandidate,
    ) -> bool:
        if isinstance(candidate, SpherePlaneAdaptiveCandidate):
            return first_body == candidate.sphere_runtime_body_id or second_body == candidate.sphere_runtime_body_id
        return {first_body, second_body} == {candidate.body_a_id, candidate.body_b_id}

    def _update_state(
        self,
        *,
        state_before: ContactMotionState,
        selected_candidate_id: str | None,
        pre_active: tuple[str, ...],
        final_active: tuple[str, ...],
        observed_active: bool,
        prediction: CollisionPrediction | None,
        final_result: SimulationStepResult,
    ) -> ContactMotionState:
        active_candidate = selected_candidate_id or (final_active[0] if final_active else None) or self._active_candidate_id
        if active_candidate is not None:
            self._active_candidate_id = active_candidate
        if final_active:
            candidate_id = final_active[0]
            if self._is_resting(candidate_id, final_result):
                self._resting_counts[candidate_id] = self._resting_counts.get(candidate_id, 0) + 1
                if self._resting_counts[candidate_id] >= self._config.resting_window_macro_steps:
                    return ContactMotionState.RESTING
            else:
                self._resting_counts[candidate_id] = 0
            return ContactMotionState.IMPACTING
        if observed_active or pre_active:
            self._separating_remaining = self._config.separating_hold_macro_steps
            return ContactMotionState.SEPARATING
        if state_before is ContactMotionState.SEPARATING:
            if self._separating_remaining > 0:
                self._separating_remaining -= 1
                return ContactMotionState.SEPARATING
            if prediction is not None:
                return ContactMotionState.APPROACHING
            self._active_candidate_id = None
            return ContactMotionState.FREE
        if prediction is not None:
            return ContactMotionState.APPROACHING
        if state_before is ContactMotionState.RESTING:
            return ContactMotionState.RESTING if final_active else ContactMotionState.FREE
        return ContactMotionState.FREE

    def _is_resting(self, candidate_id: str, result: SimulationStepResult) -> bool:
        candidate = self._candidate_by_id[candidate_id]
        if isinstance(candidate, SpherePlaneAdaptiveCandidate):
            state = result.get_body_state(candidate.sphere_runtime_body_id)
            normal_speed = abs(_dot(_subtract(state.linear_velocity, candidate.plane.linear_velocity), candidate.plane.normal))
            return (
                normal_speed <= self._config.resting_normal_speed_threshold
                and _norm(state.linear_velocity) <= self._config.resting_linear_speed_threshold
                and _norm(state.angular_velocity) <= self._config.resting_angular_speed_threshold
            )
        if isinstance(candidate, ConservativePrimitiveAdaptiveCandidate):
            first = result.get_body_state(candidate.body_a_id)
            second = result.get_body_state(candidate.body_b_id)
            relative_velocity = _subtract(second.linear_velocity, first.linear_velocity)
            return (
                _norm(relative_velocity) <= self._config.resting_linear_speed_threshold
                and _norm(first.angular_velocity) <= self._config.resting_angular_speed_threshold
                and _norm(second.angular_velocity) <= self._config.resting_angular_speed_threshold
            )
        first = result.get_body_state(candidate.body_a_id)
        second = result.get_body_state(candidate.body_b_id)
        offset = _subtract(second.position, first.position)
        normal = _normalize(offset)
        relative_velocity = _subtract(second.linear_velocity, first.linear_velocity)
        normal_speed = abs(_dot(relative_velocity, normal))
        return (
            normal_speed <= self._config.resting_normal_speed_threshold
            and _norm(first.linear_velocity) <= self._config.resting_linear_speed_threshold
            and _norm(second.linear_velocity) <= self._config.resting_linear_speed_threshold
            and _norm(first.angular_velocity) <= self._config.resting_angular_speed_threshold
            and _norm(second.angular_velocity) <= self._config.resting_angular_speed_threshold
        )

    def _decision_reason(
        self,
        state_before: ContactMotionState,
        selected_candidate_id: str | None,
        pre_active: tuple[str, ...],
        recommendation: SubstepRecommendation | None,
    ) -> str:
        if selected_candidate_id is not None:
            return "prediction selected candidate with smallest actual substep timestep"
        if pre_active:
            return "active contact observed before step; using candidate solver recommendation"
        if recommendation is not None and state_before in {
            ContactMotionState.APPROACHING,
            ContactMotionState.IMPACTING,
            ContactMotionState.SEPARATING,
        }:
            return "continuing cached impact/separation substep recommendation"
        if state_before is ContactMotionState.RESTING:
            return "resting contact uses macro timestep"
        return "no active or predicted impact; using macro timestep"


@dataclass(frozen=True)
class _CandidateEstimate:
    prediction_candidate_id: str
    estimate: SolverCollisionEstimate


def _validate_candidate_id(value: str) -> None:
    _non_empty_string(value, field_name="candidate_id", error_type=PhysicsValidationError)


def _validate_body_id(value: str, field_name: str) -> None:
    _non_empty_string(value, field_name=field_name, error_type=PhysicsValidationError)


def _validate_contact_params(value: MuJoCoContactSolverParams) -> None:
    if not isinstance(value, MuJoCoContactSolverParams):
        raise PhysicsValidationError(f"contact_params must be MuJoCoContactSolverParams; actual value={value!r}")


def _vector3(value: Vector3, field_name: str) -> Vector3:
    if len(value) != 3:
        raise PhysicsValidationError(f"{field_name} must be a 3D vector; actual value={value!r}")
    return tuple(
        _finite_float(component, field_name=f"{field_name}[{index}]", error_type=PhysicsValidationError)
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _subtract(first: Vector3, second: Vector3) -> Vector3:
    return tuple(first[index] - second[index] for index in range(3))  # type: ignore[return-value]


def _add(first: Vector3, second: Vector3) -> Vector3:
    return tuple(first[index] + second[index] for index in range(3))  # type: ignore[return-value]


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return tuple(value * scalar for value in vector)  # type: ignore[return-value]


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(first[index] * second[index] for index in range(3))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector3) -> Vector3:
    length = _norm(vector)
    if length <= 1.0e-12:
        return (1.0, 0.0, 0.0)
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def _point_in_bounds(point: Vector3, bounds: FinitePlaneBounds) -> bool:
    offset = _subtract(point, bounds.center)
    u = _dot(offset, bounds.axis_u)
    v = _dot(offset, bounds.axis_v)
    return abs(u) <= bounds.half_extent_u + 1.0e-12 and abs(v) <= bounds.half_extent_v + 1.0e-12


def _predict_conservative_bounding_spheres(
    *,
    body_a_position: Vector3,
    body_a_velocity: Vector3,
    body_a_angular_velocity: Vector3,
    radius_a: float,
    body_b_position: Vector3,
    body_b_velocity: Vector3,
    body_b_angular_velocity: Vector3,
    radius_b: float,
    prediction_horizon: float,
) -> CollisionPrediction | None:
    offset = _subtract(body_b_position, body_a_position)
    distance = _norm(offset)
    normal = _normalize(offset)
    gap = distance - radius_a - radius_b
    relative_velocity = _subtract(body_b_velocity, body_a_velocity)
    closing_speed = max(0.0, -_dot(relative_velocity, normal))
    angular_speed_bound = _norm(body_a_angular_velocity) * radius_a + _norm(body_b_angular_velocity) * radius_b
    conservative_speed = closing_speed + angular_speed_bound
    if gap <= 0.0:
        return CollisionPrediction("conservative_bounding_sphere", 0.0, gap, conservative_speed, normal)
    if conservative_speed <= 1.0e-12:
        return None
    time_to_contact = gap / conservative_speed
    if time_to_contact > prediction_horizon:
        return None
    return CollisionPrediction("conservative_bounding_sphere", time_to_contact, gap, conservative_speed, normal)
