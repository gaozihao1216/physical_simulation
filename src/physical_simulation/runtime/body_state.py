"""Runtime rigid body state data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from physical_simulation.validation.asset_validator import _as_float_tuple, _non_empty_string
from physical_simulation.validation.errors import InvalidRuntimeStateError


@dataclass(frozen=True)
class RigidBodyState:
    """Runtime state for one body, separate from rigid body specification."""

    body_id: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "body_id",
            _non_empty_string(
                self.body_id,
                field_name="body_id",
                error_type=InvalidRuntimeStateError,
            ),
        )
        object.__setattr__(
            self,
            "position",
            _as_float_tuple(
                self.position,
                field_name="position",
                length=3,
                error_type=InvalidRuntimeStateError,
            ),
        )
        rotation = _as_float_tuple(
            self.rotation,
            field_name="rotation",
            length=4,
            error_type=InvalidRuntimeStateError,
        )
        norm = math.sqrt(sum(component * component for component in rotation))
        if norm <= 1.0e-12:
            raise InvalidRuntimeStateError(
                f"rotation quaternion norm must be > 1e-12; actual value={self.rotation!r}"
            )
        object.__setattr__(self, "rotation", tuple(component / norm for component in rotation))
        object.__setattr__(
            self,
            "linear_velocity",
            _as_float_tuple(
                self.linear_velocity,
                field_name="linear_velocity",
                length=3,
                error_type=InvalidRuntimeStateError,
            ),
        )
        object.__setattr__(
            self,
            "angular_velocity",
            _as_float_tuple(
                self.angular_velocity,
                field_name="angular_velocity",
                length=3,
                error_type=InvalidRuntimeStateError,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state to a JSON-compatible dictionary."""
        return {
            "body_id": self.body_id,
            "position": list(self.position),
            "rotation": list(self.rotation),
            "linear_velocity": list(self.linear_velocity),
            "angular_velocity": list(self.angular_velocity),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RigidBodyState":
        """Deserialize body state from a dictionary."""
        if not isinstance(data, dict):
            raise InvalidRuntimeStateError(f"body state data must be a dict; actual value={data!r}")
        return cls(
            body_id=data.get("body_id"),
            position=tuple(data.get("position", ())),
            rotation=tuple(data.get("rotation", ())),
            linear_velocity=tuple(data.get("linear_velocity", ())),
            angular_velocity=tuple(data.get("angular_velocity", ())),
        )
