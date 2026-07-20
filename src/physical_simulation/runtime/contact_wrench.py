"""Runtime contact wrench data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from physical_simulation.runtime.contact import ContactPoint
from physical_simulation.validation.asset_validator import _as_float_tuple, _finite_float
from physical_simulation.validation.errors import InvalidRuntimeStateError


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class ContactWrench:
    """Solver-produced wrench for one runtime contact point."""

    contact: ContactPoint
    force_on_body_a_world: Vector3
    contact_torque_on_body_a_world: Vector3
    force_on_body_b_world: Vector3
    contact_torque_on_body_b_world: Vector3
    normal_force_magnitude: float
    tangential_force_magnitude: float

    def __post_init__(self) -> None:
        if not isinstance(self.contact, ContactPoint):
            raise InvalidRuntimeStateError(
                f"contact must be ContactPoint; actual type={type(self.contact).__name__}, value={self.contact!r}"
            )
        for field_name in (
            "force_on_body_a_world",
            "contact_torque_on_body_a_world",
            "force_on_body_b_world",
            "contact_torque_on_body_b_world",
        ):
            object.__setattr__(
                self,
                field_name,
                _as_float_tuple(
                    getattr(self, field_name),
                    field_name=field_name,
                    length=3,
                    error_type=InvalidRuntimeStateError,
                ),
            )
        object.__setattr__(
            self,
            "normal_force_magnitude",
            _finite_float(
                self.normal_force_magnitude,
                field_name="normal_force_magnitude",
                minimum=0.0,
                error_type=InvalidRuntimeStateError,
            ),
        )
        object.__setattr__(
            self,
            "tangential_force_magnitude",
            _finite_float(
                self.tangential_force_magnitude,
                field_name="tangential_force_magnitude",
                minimum=0.0,
                error_type=InvalidRuntimeStateError,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the contact wrench to a JSON-compatible dictionary."""
        return {
            "contact": self.contact.to_dict(),
            "force_on_body_a_world": list(self.force_on_body_a_world),
            "contact_torque_on_body_a_world": list(self.contact_torque_on_body_a_world),
            "force_on_body_b_world": list(self.force_on_body_b_world),
            "contact_torque_on_body_b_world": list(self.contact_torque_on_body_b_world),
            "normal_force_magnitude": self.normal_force_magnitude,
            "tangential_force_magnitude": self.tangential_force_magnitude,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContactWrench":
        """Deserialize a contact wrench from a dictionary."""
        if not isinstance(data, dict):
            raise InvalidRuntimeStateError(f"contact wrench data must be a dict; actual value={data!r}")
        return cls(
            contact=ContactPoint.from_dict(data.get("contact", {})),
            force_on_body_a_world=tuple(data.get("force_on_body_a_world", ())),
            contact_torque_on_body_a_world=tuple(data.get("contact_torque_on_body_a_world", ())),
            force_on_body_b_world=tuple(data.get("force_on_body_b_world", ())),
            contact_torque_on_body_b_world=tuple(data.get("contact_torque_on_body_b_world", ())),
            normal_force_magnitude=data.get("normal_force_magnitude"),
            tangential_force_magnitude=data.get("tangential_force_magnitude"),
        )
