"""Physical material specifications for Physics IR assets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from physical_simulation.validation.asset_validator import _finite_float, _non_empty_string
from physical_simulation.validation.errors import PhysicsValidationError


@dataclass(frozen=True)
class PhysicsMaterialSpec:
    """Friction, restitution, and optional density for a physical material."""

    material_id: str
    static_friction: float = 0.5
    dynamic_friction: float = 0.4
    restitution: float = 0.0
    density: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "material_id",
            _non_empty_string(
                self.material_id,
                field_name="material_id",
                error_type=PhysicsValidationError,
            ),
        )
        object.__setattr__(
            self,
            "static_friction",
            _finite_float(
                self.static_friction,
                field_name="static_friction",
                minimum=0.0,
                error_type=PhysicsValidationError,
            ),
        )
        object.__setattr__(
            self,
            "dynamic_friction",
            _finite_float(
                self.dynamic_friction,
                field_name="dynamic_friction",
                minimum=0.0,
                error_type=PhysicsValidationError,
            ),
        )
        object.__setattr__(
            self,
            "restitution",
            _finite_float(
                self.restitution,
                field_name="restitution",
                minimum=0.0,
                maximum=1.0,
                error_type=PhysicsValidationError,
            ),
        )
        if self.density is not None:
            object.__setattr__(
                self,
                "density",
                _finite_float(
                    self.density,
                    field_name="density",
                    minimum=0.0,
                    strict_minimum=True,
                    error_type=PhysicsValidationError,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the material to a JSON-compatible dictionary."""
        return {
            "material_id": self.material_id,
            "static_friction": self.static_friction,
            "dynamic_friction": self.dynamic_friction,
            "restitution": self.restitution,
            "density": self.density,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhysicsMaterialSpec":
        """Deserialize a material from a dictionary."""
        if not isinstance(data, dict):
            raise PhysicsValidationError(f"material data must be a dict; actual value={data!r}")
        return cls(
            material_id=data.get("material_id"),
            static_friction=data.get("static_friction", 0.5),
            dynamic_friction=data.get("dynamic_friction", 0.4),
            restitution=data.get("restitution", 0.0),
            density=data.get("density"),
        )


DEFAULT_MATERIAL = PhysicsMaterialSpec(
    material_id="default",
    static_friction=0.5,
    dynamic_friction=0.4,
    restitution=0.0,
    density=1000.0,
)
