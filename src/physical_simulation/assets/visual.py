"""Visual asset specifications for Physics IR rigid bodies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from physical_simulation.assets.geometry import GeometrySpec, geometry_from_dict
from physical_simulation.assets.transform import Transform
from physical_simulation.validation.asset_validator import (
    _non_empty_string,
    validate_geometry,
    validate_transform,
)
from physical_simulation.validation.errors import PhysicsValidationError


@dataclass(frozen=True)
class VisualSpec:
    """Visual representation of a rigid body part.

    Visual geometry is intentionally separate from collision geometry.
    """

    visual_id: str
    geometry: GeometrySpec
    local_transform: Transform = Transform.identity()
    material_name: Optional[str] = None
    visible: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "visual_id",
            _non_empty_string(
                self.visual_id,
                field_name="visual_id",
                error_type=PhysicsValidationError,
            ),
        )
        validate_geometry(self.geometry)
        validate_transform(self.local_transform)
        if self.material_name is not None and not isinstance(self.material_name, str):
            raise PhysicsValidationError(
                f"material_name must be str or None; actual value={self.material_name!r}"
            )
        if not isinstance(self.visible, bool):
            raise PhysicsValidationError(f"visible must be bool; actual value={self.visible!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the visual specification to a JSON-compatible dictionary."""
        return {
            "visual_id": self.visual_id,
            "geometry": self.geometry.to_dict(),
            "local_transform": self.local_transform.to_dict(),
            "material_name": self.material_name,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualSpec":
        """Deserialize a visual specification from a dictionary."""
        if not isinstance(data, dict):
            raise PhysicsValidationError(f"visual data must be a dict; actual value={data!r}")
        return cls(
            visual_id=data.get("visual_id"),
            geometry=geometry_from_dict(data.get("geometry")),
            local_transform=Transform.from_dict(data.get("local_transform", {})),
            material_name=data.get("material_name"),
            visible=data.get("visible", True),
        )
