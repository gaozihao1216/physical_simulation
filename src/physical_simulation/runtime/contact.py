"""Runtime contact point data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from physical_simulation.validation.asset_validator import (
    _as_float_tuple,
    _finite_float,
    _non_empty_string,
)
from physical_simulation.validation.errors import InvalidRuntimeStateError


@dataclass(frozen=True)
class ContactPoint:
    """Runtime contact point.

    ``normal`` points from ``body_a`` toward ``body_b``.
    """

    body_a: str
    body_b: str
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    penetration_depth: float
    normal_force: Optional[float] = None
    tangential_force: Optional[tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        body_a = _non_empty_string(
            self.body_a,
            field_name="body_a",
            error_type=InvalidRuntimeStateError,
        )
        body_b = _non_empty_string(
            self.body_b,
            field_name="body_b",
            error_type=InvalidRuntimeStateError,
        )
        if body_a == body_b:
            raise InvalidRuntimeStateError(
                f"body_a and body_b must be different; actual value={body_a!r}"
            )
        object.__setattr__(self, "body_a", body_a)
        object.__setattr__(self, "body_b", body_b)
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
        normal = _as_float_tuple(
            self.normal,
            field_name="normal",
            length=3,
            error_type=InvalidRuntimeStateError,
        )
        norm = math.sqrt(sum(component * component for component in normal))
        if norm <= 1.0e-12:
            raise InvalidRuntimeStateError(
                f"normal vector norm must be > 1e-12; actual value={self.normal!r}"
            )
        object.__setattr__(self, "normal", tuple(component / norm for component in normal))
        object.__setattr__(
            self,
            "penetration_depth",
            _finite_float(
                self.penetration_depth,
                field_name="penetration_depth",
                minimum=0.0,
                error_type=InvalidRuntimeStateError,
            ),
        )
        if self.normal_force is not None:
            object.__setattr__(
                self,
                "normal_force",
                _finite_float(
                    self.normal_force,
                    field_name="normal_force",
                    minimum=0.0,
                    error_type=InvalidRuntimeStateError,
                ),
            )
        if self.tangential_force is not None:
            object.__setattr__(
                self,
                "tangential_force",
                _as_float_tuple(
                    self.tangential_force,
                    field_name="tangential_force",
                    length=3,
                    error_type=InvalidRuntimeStateError,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the contact point to a JSON-compatible dictionary."""
        return {
            "body_a": self.body_a,
            "body_b": self.body_b,
            "position": list(self.position),
            "normal": list(self.normal),
            "penetration_depth": self.penetration_depth,
            "normal_force": self.normal_force,
            "tangential_force": None if self.tangential_force is None else list(self.tangential_force),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContactPoint":
        """Deserialize contact point from a dictionary."""
        if not isinstance(data, dict):
            raise InvalidRuntimeStateError(f"contact data must be a dict; actual value={data!r}")
        tangential_force = data.get("tangential_force")
        return cls(
            body_a=data.get("body_a"),
            body_b=data.get("body_b"),
            position=tuple(data.get("position", ())),
            normal=tuple(data.get("normal", ())),
            penetration_depth=data.get("penetration_depth"),
            normal_force=data.get("normal_force"),
            tangential_force=None if tangential_force is None else tuple(tangential_force),
        )
