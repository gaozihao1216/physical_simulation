"""Collision shape specifications for Physics IR rigid bodies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from physical_simulation.assets.geometry import GeometrySpec, geometry_from_dict
from physical_simulation.assets.transform import Transform
from physical_simulation.validation.asset_validator import (
    _non_empty_string,
    validate_geometry,
    validate_transform,
)
from physical_simulation.validation.errors import PhysicsValidationError


@dataclass(frozen=True)
class ColliderSpec:
    """Collision representation of a rigid body part."""

    collider_id: str
    geometry: GeometrySpec
    local_transform: Transform = Transform.identity()
    material_id: str = "default"
    collision_group: int = 1
    collision_mask: int = -1
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "collider_id",
            _non_empty_string(
                self.collider_id,
                field_name="collider_id",
                error_type=PhysicsValidationError,
            ),
        )
        object.__setattr__(
            self,
            "material_id",
            _non_empty_string(
                self.material_id,
                field_name="material_id",
                error_type=PhysicsValidationError,
            ),
        )
        validate_geometry(self.geometry)
        validate_transform(self.local_transform)
        from physical_simulation.assets.scale_baking import is_unit_scale

        if not is_unit_scale(self.local_transform.scale):
            raise PhysicsValidationError(
                "local_transform.scale must be unit scale for ColliderSpec; "
                f"actual value={self.local_transform.scale!r}; "
                "call bake_transform_scale() before creating physical colliders"
            )
        if not isinstance(self.collision_group, int) or isinstance(self.collision_group, bool):
            raise PhysicsValidationError(
                f"collision_group must be int; actual value={self.collision_group!r}"
            )
        if not isinstance(self.collision_mask, int) or isinstance(self.collision_mask, bool):
            raise PhysicsValidationError(
                f"collision_mask must be int; actual value={self.collision_mask!r}"
            )
        if not isinstance(self.enabled, bool):
            raise PhysicsValidationError(f"enabled must be bool; actual value={self.enabled!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the collider specification to a JSON-compatible dictionary."""
        return {
            "collider_id": self.collider_id,
            "geometry": self.geometry.to_dict(),
            "local_transform": self.local_transform.to_dict(),
            "material_id": self.material_id,
            "collision_group": self.collision_group,
            "collision_mask": self.collision_mask,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColliderSpec":
        """Deserialize a collider specification from a dictionary."""
        if not isinstance(data, dict):
            raise PhysicsValidationError(f"collider data must be a dict; actual value={data!r}")
        return cls(
            collider_id=data.get("collider_id"),
            geometry=geometry_from_dict(data.get("geometry")),
            local_transform=Transform.from_dict(data.get("local_transform", {})),
            material_id=data.get("material_id", "default"),
            collision_group=data.get("collision_group", 1),
            collision_mask=data.get("collision_mask", -1),
            enabled=data.get("enabled", True),
        )
