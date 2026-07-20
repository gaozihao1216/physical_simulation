"""Spatial transform representation for Physics IR assets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from physical_simulation.validation.asset_validator import _as_float_tuple
from physical_simulation.validation.errors import PhysicsValidationError


@dataclass(frozen=True)
class Transform:
    """Position, quaternion rotation, and scale in the project coordinate system.

    Units are meters for position. Rotations use quaternions in ``(w, x, y, z)`` order.
    """

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        position = _as_float_tuple(
            self.position,
            field_name="position",
            length=3,
            error_type=PhysicsValidationError,
        )
        rotation = _as_float_tuple(
            self.rotation,
            field_name="rotation",
            length=4,
            error_type=PhysicsValidationError,
        )
        scale = _as_float_tuple(
            self.scale,
            field_name="scale",
            length=3,
            strictly_positive=True,
            error_type=PhysicsValidationError,
        )
        norm = math.sqrt(sum(component * component for component in rotation))
        if norm <= 1.0e-12:
            raise PhysicsValidationError(
                f"rotation quaternion norm must be > 1e-12; actual value={self.rotation!r}"
            )
        normalized_rotation = tuple(component / norm for component in rotation)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "rotation", normalized_rotation)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def identity(cls) -> "Transform":
        """Return an identity transform."""
        return cls()

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize the transform to a JSON-compatible dictionary."""
        return {
            "position": list(self.position),
            "rotation": list(self.rotation),
            "scale": list(self.scale),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transform":
        """Deserialize a transform from a dictionary."""
        if not isinstance(data, dict):
            raise PhysicsValidationError(f"transform data must be a dict; actual value={data!r}")
        return cls(
            position=tuple(data.get("position", (0.0, 0.0, 0.0))),
            rotation=tuple(data.get("rotation", (1.0, 0.0, 0.0, 0.0))),
            scale=tuple(data.get("scale", (1.0, 1.0, 1.0))),
        )
