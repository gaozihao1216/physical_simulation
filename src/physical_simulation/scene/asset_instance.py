"""Scene instance of a reusable physics asset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from physical_simulation.assets.physics_asset import PhysicsAssetSpec
from physical_simulation.assets.transform import Transform
from physical_simulation.validation.asset_validator import (
    _non_empty_string,
    validate_physics_asset,
    validate_transform,
)
from physical_simulation.validation.errors import InvalidPhysicsSceneError


@dataclass(frozen=True)
class AssetInstanceSpec:
    """Placement of a reusable physics asset in a scene."""

    instance_id: str
    asset: PhysicsAssetSpec
    transform: Transform = Transform.identity()
    fixed_base: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instance_id",
            _non_empty_string(
                self.instance_id,
                field_name="instance_id",
                error_type=InvalidPhysicsSceneError,
            ),
        )
        validate_physics_asset(self.asset)
        validate_transform(self.transform)
        from physical_simulation.assets.scale_baking import is_unit_scale

        if not is_unit_scale(self.transform.scale):
            raise InvalidPhysicsSceneError(
                "transform.scale must be unit scale for AssetInstanceSpec; "
                f"actual value={self.transform.scale!r}; "
                "bake scale into PhysicsAssetSpec before creating scene instances"
            )
        if not isinstance(self.fixed_base, bool):
            raise InvalidPhysicsSceneError(
                f"fixed_base must be bool; actual value={self.fixed_base!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the asset instance to a JSON-compatible dictionary."""
        return {
            "instance_id": self.instance_id,
            "asset": self.asset.to_dict(),
            "transform": self.transform.to_dict(),
            "fixed_base": self.fixed_base,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssetInstanceSpec":
        """Deserialize an asset instance from a dictionary."""
        if not isinstance(data, dict):
            raise InvalidPhysicsSceneError(f"instance data must be a dict; actual value={data!r}")
        return cls(
            instance_id=data.get("instance_id"),
            asset=PhysicsAssetSpec.from_dict(data.get("asset")),
            transform=Transform.from_dict(data.get("transform", {})),
            fixed_base=data.get("fixed_base", False),
        )
