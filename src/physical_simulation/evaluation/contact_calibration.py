"""Contact behavior calibration helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable

from physical_simulation.backends.base import PhysicsBackend
from physical_simulation.validation.asset_validator import _finite_float


@dataclass(frozen=True)
class ReferenceRestitutionTarget:
    """Backend-independent target used for restitution calibration experiments."""

    restitution: float
    reference_impact_speed: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "restitution",
            _finite_float(self.restitution, field_name="restitution", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "reference_impact_speed",
            _finite_float(
                self.reference_impact_speed,
                field_name="reference_impact_speed",
                minimum=0.0,
                strict_minimum=True,
            ),
        )


class RestitutionOutcome(Enum):
    """Outcome classification for a restitution measurement rollout."""

    REBOUNDED = "rebounded"
    SETTLED_IN_CONTACT = "settled_in_contact"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class RestitutionMeasurement:
    """Measured restitution-like response from a standard drop experiment."""

    runtime_body_id: str
    outcome: RestitutionOutcome
    impact_speed: float
    rebound_speed: float | None
    measured_restitution: float | None
    contact_start_step: int
    contact_end_step: int | None
    contact_duration_steps: int | None
    contact_duration_seconds: float | None
    maximum_penetration_depth: float
    normalized_penetration: float | None
    observed_contact_steps: int


@dataclass(frozen=True)
class RestitutionSweepSample:
    """One initial-height sample from a restitution sweep."""

    initial_height: float
    measurement: RestitutionMeasurement


def measure_restitution(
    backend: PhysicsBackend,
    runtime_body_id: str,
    *,
    max_steps: int,
    contact_force_threshold: float = 0.0,
    rebound_velocity_threshold: float = 1.0e-6,
    settling_linear_speed_threshold: float = 0.02,
    settling_angular_speed_threshold: float = 0.05,
    settling_window_steps: int = 120,
    characteristic_length: float | None = None,
) -> RestitutionMeasurement:
    """Measure rebound behavior from one backend reset and fixed-step rollout."""
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError(f"max_steps must be a positive int; actual value={max_steps!r}")
    contact_force_threshold = _finite_float(
        contact_force_threshold,
        field_name="contact_force_threshold",
        minimum=0.0,
    )
    rebound_velocity_threshold = _finite_float(
        rebound_velocity_threshold,
        field_name="rebound_velocity_threshold",
        minimum=0.0,
    )
    settling_linear_speed_threshold = _finite_float(
        settling_linear_speed_threshold,
        field_name="settling_linear_speed_threshold",
        minimum=0.0,
    )
    settling_angular_speed_threshold = _finite_float(
        settling_angular_speed_threshold,
        field_name="settling_angular_speed_threshold",
        minimum=0.0,
    )
    if (
        not isinstance(settling_window_steps, int)
        or isinstance(settling_window_steps, bool)
        or settling_window_steps < 1
    ):
        raise ValueError(f"settling_window_steps must be a positive int; actual value={settling_window_steps!r}")
    if characteristic_length is not None:
        characteristic_length = _finite_float(
            characteristic_length,
            field_name="characteristic_length",
            minimum=0.0,
            strict_minimum=True,
        )

    result = backend.reset()
    last_downward_speed = 0.0
    impact_speed = 0.0
    contact_start_step: int | None = None
    contact_end_step: int | None = None
    last_contact_step: int | None = None
    previous_time = result.time
    timestep: float | None = None
    maximum_penetration_depth = 0.0
    observed_contact_steps = 0
    contact_window_speeds: list[tuple[float, float]] = []

    for _ in range(max_steps + 1):
        state = result.get_body_state(runtime_body_id)
        vertical_speed = float(state.linear_velocity[2])
        linear_speed = _vector_norm(state.linear_velocity)
        angular_speed = _vector_norm(state.angular_velocity)
        contacts = tuple(
            contact
            for contact in result.contacts
            if contact.body_a == runtime_body_id or contact.body_b == runtime_body_id
        )
        if result.step_index > 0 and timestep is None:
            timestep = result.time - previous_time

        if contact_start_step is None and vertical_speed < 0.0:
            last_downward_speed = -vertical_speed

        if contacts and _contact_force_exceeds_threshold(backend, runtime_body_id, contact_force_threshold):
            if contact_start_step is None:
                contact_start_step = result.step_index
                impact_speed = last_downward_speed
            last_contact_step = result.step_index
            observed_contact_steps += 1
            contact_window_speeds.append((linear_speed, angular_speed))
            contact_window_speeds = contact_window_speeds[-settling_window_steps:]
            maximum_penetration_depth = max(
                maximum_penetration_depth,
                max(contact.penetration_depth for contact in contacts),
            )
            if (
                len(contact_window_speeds) >= settling_window_steps
                and all(speed <= settling_linear_speed_threshold for speed, _ in contact_window_speeds)
                and all(speed <= settling_angular_speed_threshold for _, speed in contact_window_speeds)
            ):
                return _build_measurement(
                    runtime_body_id=runtime_body_id,
                    outcome=RestitutionOutcome.SETTLED_IN_CONTACT,
                    impact_speed=impact_speed,
                    rebound_speed=0.0,
                    measured_restitution=0.0,
                    contact_start_step=contact_start_step,
                    contact_end_step=None,
                    timestep=timestep,
                    maximum_penetration_depth=maximum_penetration_depth,
                    characteristic_length=characteristic_length,
                    observed_contact_steps=observed_contact_steps,
                )
        elif contact_start_step is not None and contact_end_step is None and last_contact_step is not None:
            contact_end_step = last_contact_step

        if contact_end_step is not None and vertical_speed > rebound_velocity_threshold and not contacts:
            rebound_speed = vertical_speed
            measured_restitution = rebound_speed / impact_speed if impact_speed > 0.0 else None
            return _build_measurement(
                runtime_body_id=runtime_body_id,
                outcome=RestitutionOutcome.REBOUNDED,
                impact_speed=impact_speed,
                rebound_speed=rebound_speed,
                measured_restitution=measured_restitution,
                contact_start_step=contact_start_step,
                contact_end_step=contact_end_step,
                timestep=timestep,
                maximum_penetration_depth=maximum_penetration_depth,
                characteristic_length=characteristic_length,
                observed_contact_steps=observed_contact_steps,
            )

        if result.step_index >= max_steps:
            break
        previous_time = result.time
        result = backend.step()

    if contact_start_step is None:
        contact_start_step = -1
    elif contact_end_step is None and last_contact_step is not None and not contacts:
        contact_end_step = last_contact_step
    return _build_measurement(
        runtime_body_id=runtime_body_id,
        outcome=RestitutionOutcome.TIMEOUT,
        impact_speed=impact_speed,
        rebound_speed=None,
        measured_restitution=None,
        contact_start_step=contact_start_step,
        contact_end_step=contact_end_step,
        timestep=timestep,
        maximum_penetration_depth=maximum_penetration_depth,
        characteristic_length=characteristic_length,
        observed_contact_steps=observed_contact_steps,
    )


def measure_restitution_sweep(
    scene_factory: Callable[[float], object],
    backend_factory: Callable[[], PhysicsBackend],
    runtime_body_id: str,
    *,
    initial_heights: Iterable[float],
    max_steps: int,
    characteristic_length: float | None = None,
    **measurement_kwargs: object,
) -> tuple[RestitutionSweepSample, ...]:
    """Run restitution measurements for several initial heights sorted by measured impact speed."""
    samples: list[RestitutionSweepSample] = []
    for initial_height in initial_heights:
        height = _finite_float(
            initial_height,
            field_name="initial_height",
            minimum=0.0,
            strict_minimum=True,
        )
        backend = backend_factory()
        backend.load_scene(scene_factory(height))  # type: ignore[arg-type]
        try:
            measurement = measure_restitution(
                backend,
                runtime_body_id,
                max_steps=max_steps,
                characteristic_length=characteristic_length,
                **measurement_kwargs,
            )
        finally:
            backend.close()
        samples.append(RestitutionSweepSample(initial_height=height, measurement=measurement))
    return tuple(sorted(samples, key=lambda sample: sample.measurement.impact_speed))


def _build_measurement(
    *,
    runtime_body_id: str,
    outcome: RestitutionOutcome,
    impact_speed: float,
    rebound_speed: float | None,
    measured_restitution: float | None,
    contact_start_step: int,
    contact_end_step: int | None,
    timestep: float | None,
    maximum_penetration_depth: float,
    characteristic_length: float | None,
    observed_contact_steps: int,
) -> RestitutionMeasurement:
    contact_duration_steps = (
        None
        if contact_end_step is None or contact_start_step < 0
        else max(0, contact_end_step - contact_start_step + 1)
    )
    contact_duration_seconds = (
        None
        if contact_duration_steps is None or timestep is None
        else contact_duration_steps * timestep
    )
    normalized_penetration = (
        None
        if characteristic_length is None
        else maximum_penetration_depth / characteristic_length
    )
    if measured_restitution is not None and (not math.isfinite(measured_restitution) or measured_restitution < 0.0):
        measured_restitution = None
    return RestitutionMeasurement(
        runtime_body_id=runtime_body_id,
        outcome=outcome,
        impact_speed=impact_speed,
        rebound_speed=rebound_speed,
        measured_restitution=measured_restitution,
        contact_start_step=contact_start_step,
        contact_end_step=contact_end_step,
        contact_duration_steps=contact_duration_steps,
        contact_duration_seconds=contact_duration_seconds,
        maximum_penetration_depth=maximum_penetration_depth,
        normalized_penetration=normalized_penetration,
        observed_contact_steps=observed_contact_steps,
    )


def _contact_force_exceeds_threshold(
    backend: PhysicsBackend,
    runtime_body_id: str,
    threshold: float,
) -> bool:
    if threshold <= 0.0:
        return True
    try:
        for wrench in backend.get_body_contact_wrenches():
            if wrench.body_id == runtime_body_id and _vector_norm(wrench.net_force_world) >= threshold:
                return True
    except Exception:
        return True
    return False


def _vector_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))
