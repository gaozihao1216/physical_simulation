"""Contact behavior calibration helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

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


@dataclass(frozen=True)
class RestitutionMeasurement:
    """Measured restitution-like response from a standard drop experiment."""

    runtime_body_id: str
    impact_speed: float
    rebound_speed: float
    measured_restitution: float
    contact_start_step: int | None
    contact_end_step: int | None
    maximum_penetration_depth: float

    @property
    def contact_duration_steps(self) -> int:
        """Return inclusive contact duration in simulation steps."""
        if self.contact_start_step is None or self.contact_end_step is None:
            return 0
        return max(0, self.contact_end_step - self.contact_start_step + 1)


def measure_restitution(
    backend: PhysicsBackend,
    runtime_body_id: str,
    *,
    max_steps: int,
) -> RestitutionMeasurement:
    """Measure rebound behavior from one backend reset and fixed-step rollout."""
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError(f"max_steps must be a positive int; actual value={max_steps!r}")

    result = backend.reset()
    last_downward_speed = 0.0
    impact_speed = 0.0
    rebound_speed = 0.0
    contact_start_step: int | None = None
    contact_end_step: int | None = None
    last_contact_step: int | None = None
    maximum_penetration_depth = 0.0

    for _ in range(max_steps + 1):
        state = result.get_body_state(runtime_body_id)
        vertical_speed = float(state.linear_velocity[2])
        contacts = tuple(
            contact
            for contact in result.contacts
            if contact.body_a == runtime_body_id or contact.body_b == runtime_body_id
        )

        if contact_start_step is None and vertical_speed < 0.0:
            last_downward_speed = -vertical_speed

        if contacts:
            if contact_start_step is None:
                contact_start_step = result.step_index
                impact_speed = last_downward_speed
            last_contact_step = result.step_index
            maximum_penetration_depth = max(
                maximum_penetration_depth,
                max(contact.penetration_depth for contact in contacts),
            )
        elif contact_start_step is not None and contact_end_step is None and last_contact_step is not None:
            contact_end_step = last_contact_step

        if contact_end_step is not None and vertical_speed > 1.0e-6:
            rebound_speed = vertical_speed
            break

        if result.step_index >= max_steps:
            break
        result = backend.step()

    if contact_start_step is not None and contact_end_step is None and last_contact_step is not None:
        contact_end_step = last_contact_step

    measured_restitution = rebound_speed / impact_speed if impact_speed > 0.0 else 0.0
    if not math.isfinite(measured_restitution) or measured_restitution < 0.0:
        measured_restitution = 0.0
    return RestitutionMeasurement(
        runtime_body_id=runtime_body_id,
        impact_speed=impact_speed,
        rebound_speed=rebound_speed,
        measured_restitution=measured_restitution,
        contact_start_step=contact_start_step,
        contact_end_step=contact_end_step,
        maximum_penetration_depth=maximum_penetration_depth,
    )
