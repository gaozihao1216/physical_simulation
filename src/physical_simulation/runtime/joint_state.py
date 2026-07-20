"""Runtime joint state data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from physical_simulation.validation.asset_validator import _as_float_tuple, _non_empty_string
from physical_simulation.validation.errors import InvalidRuntimeStateError


@dataclass(frozen=True)
class JointState:
    """Backend-independent runtime state for one joint."""

    joint_id: str
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    applied_force: Optional[tuple[float, ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "joint_id",
            _non_empty_string(
                self.joint_id,
                field_name="joint_id",
                error_type=InvalidRuntimeStateError,
            ),
        )
        position = tuple(self.position)
        velocity = tuple(self.velocity)
        if len(position) < 1:
            raise InvalidRuntimeStateError(
                f"position must contain at least one value; actual value={self.position!r}"
            )
        position = _as_float_tuple(
            position,
            field_name="position",
            length=len(position),
            error_type=InvalidRuntimeStateError,
        )
        velocity = _as_float_tuple(
            velocity,
            field_name="velocity",
            length=len(velocity),
            error_type=InvalidRuntimeStateError,
        )
        if len(position) != len(velocity):
            raise InvalidRuntimeStateError(
                f"position and velocity lengths must match; actual lengths={len(position)} and {len(velocity)}"
            )
        applied_force = None
        if self.applied_force is not None:
            applied_force = _as_float_tuple(
                tuple(self.applied_force),
                field_name="applied_force",
                length=len(position),
                error_type=InvalidRuntimeStateError,
            )
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "applied_force", applied_force)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state to a JSON-compatible dictionary."""
        return {
            "joint_id": self.joint_id,
            "position": list(self.position),
            "velocity": list(self.velocity),
            "applied_force": None if self.applied_force is None else list(self.applied_force),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JointState":
        """Deserialize joint state from a dictionary."""
        if not isinstance(data, dict):
            raise InvalidRuntimeStateError(f"joint state data must be a dict; actual value={data!r}")
        applied_force = data.get("applied_force")
        return cls(
            joint_id=data.get("joint_id"),
            position=tuple(data.get("position", ())),
            velocity=tuple(data.get("velocity", ())),
            applied_force=None if applied_force is None else tuple(applied_force),
        )
