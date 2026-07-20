"""Resting contact metrics for sampled rigid-body trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass

from physical_simulation.evaluation.metrics import quaternion_angular_distance, vector_norm
from physical_simulation.evaluation.trajectory import BodyStateSample


@dataclass(frozen=True)
class SettlingCriteria:
    """Thresholds for simple last-window settling evaluation."""

    window_steps: int = 120
    max_linear_speed: float = 0.02
    max_angular_speed: float = 0.05
    max_position_drift: float = 0.002
    max_orientation_drift: float = 0.01
    require_final_contact: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.window_steps, int) or isinstance(self.window_steps, bool) or self.window_steps < 2:
            raise ValueError(f"window_steps must be an int >= 2; actual value={self.window_steps!r}")
        for field_name in (
            "max_linear_speed",
            "max_angular_speed",
            "max_position_drift",
            "max_orientation_drift",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and >= 0; actual value={value!r}")
            object.__setattr__(self, field_name, value)
        if not isinstance(self.require_final_contact, bool):
            raise ValueError(
                f"require_final_contact must be bool; actual value={self.require_final_contact!r}"
            )


@dataclass(frozen=True)
class RestingContactMetrics:
    """Summary metrics for a body's drop/resting-contact trajectory."""

    runtime_body_id: str
    initial_height: float
    final_height: float
    minimum_height: float
    maximum_penetration_depth: float
    contact_step_count: int
    final_contact_count: int
    maximum_linear_speed: float
    maximum_angular_speed: float
    maximum_linear_speed_last_window: float
    maximum_angular_speed_last_window: float
    position_drift_last_window: float
    orientation_drift_last_window: float
    settled: bool


def evaluate_resting_contact(
    samples: tuple[BodyStateSample, ...],
    runtime_body_id: str,
    *,
    criteria: SettlingCriteria | None = None,
) -> RestingContactMetrics:
    """Evaluate simple resting-contact metrics from sampled body states."""
    selected_criteria = criteria or SettlingCriteria()
    if len(samples) < 2:
        raise ValueError("samples must contain at least two BodyStateSample values")
    for sample in samples:
        if sample.state.body_id != runtime_body_id:
            raise ValueError(
                f"sample body_id does not match runtime_body_id; "
                f"expected={runtime_body_id!r}, actual={sample.state.body_id!r}"
            )
        _validate_sample_finite(sample)

    window = samples[-min(selected_criteria.window_steps, len(samples)):]
    window_start = window[0].state
    final_contacts = _contacts_for_body(samples[-1], runtime_body_id)
    maximum_linear_speed = max(vector_norm(sample.state.linear_velocity) for sample in samples)
    maximum_angular_speed = max(vector_norm(sample.state.angular_velocity) for sample in samples)
    maximum_linear_speed_last_window = max(vector_norm(sample.state.linear_velocity) for sample in window)
    maximum_angular_speed_last_window = max(vector_norm(sample.state.angular_velocity) for sample in window)
    position_drift_last_window = max(
        vector_norm(_vector_delta(sample.state.position, window_start.position))
        for sample in window
    )
    orientation_drift_last_window = max(
        quaternion_angular_distance(sample.state.rotation, window_start.rotation)
        for sample in window
    )
    maximum_penetration_depth = max(
        (
            contact.penetration_depth
            for sample in samples
            for contact in _contacts_for_body(sample, runtime_body_id)
        ),
        default=0.0,
    )
    contact_step_count = sum(1 for sample in samples if _contacts_for_body(sample, runtime_body_id))
    final_contact_count = len(final_contacts)
    settled = (
        maximum_linear_speed_last_window <= selected_criteria.max_linear_speed
        and maximum_angular_speed_last_window <= selected_criteria.max_angular_speed
        and position_drift_last_window <= selected_criteria.max_position_drift
        and orientation_drift_last_window <= selected_criteria.max_orientation_drift
        and (final_contact_count > 0 or not selected_criteria.require_final_contact)
    )

    return RestingContactMetrics(
        runtime_body_id=runtime_body_id,
        initial_height=samples[0].state.position[2],
        final_height=samples[-1].state.position[2],
        minimum_height=min(sample.state.position[2] for sample in samples),
        maximum_penetration_depth=maximum_penetration_depth,
        contact_step_count=contact_step_count,
        final_contact_count=final_contact_count,
        maximum_linear_speed=maximum_linear_speed,
        maximum_angular_speed=maximum_angular_speed,
        maximum_linear_speed_last_window=maximum_linear_speed_last_window,
        maximum_angular_speed_last_window=maximum_angular_speed_last_window,
        position_drift_last_window=position_drift_last_window,
        orientation_drift_last_window=orientation_drift_last_window,
        settled=settled,
    )


def _contacts_for_body(sample: BodyStateSample, runtime_body_id: str):
    return tuple(
        contact
        for contact in sample.contacts
        if contact.body_a == runtime_body_id or contact.body_b == runtime_body_id
    )


def _vector_delta(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(first, second))


def _validate_sample_finite(sample: BodyStateSample) -> None:
    values = (
        sample.time,
        *sample.state.position,
        *sample.state.rotation,
        *sample.state.linear_velocity,
        *sample.state.angular_velocity,
        *(
            value
            for contact in sample.contacts
            for value in (*contact.position, *contact.normal, contact.penetration_depth)
        ),
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"sample contains non-finite values; sample={sample!r}")
