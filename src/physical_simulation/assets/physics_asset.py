"""Reusable physics asset specification."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

from physical_simulation.assets.material import DEFAULT_MATERIAL, PhysicsMaterialSpec
from physical_simulation.assets.rigid_body import RigidBodySpec
from physical_simulation.validation.asset_validator import (
    _non_empty_string,
    validate_material,
    validate_physics_asset,
    validate_rigid_body,
)
from physical_simulation.validation.errors import InvalidPhysicsAssetError, SerializationError

CURRENT_PHYSICS_ASSET_SCHEMA_VERSION = "1.0"


def _freeze_metadata(metadata: Mapping[str, str] | None) -> Mapping[str, str]:
    source = {} if metadata is None else dict(metadata)
    for key, value in source.items():
        if not isinstance(key, str) or not key.strip():
            raise InvalidPhysicsAssetError(
                f"metadata keys must be non-empty strings; actual key={key!r}"
            )
        if not isinstance(value, str):
            raise InvalidPhysicsAssetError(
                f"metadata values must be strings; actual key={key!r}, value={value!r}"
            )
    return MappingProxyType(dict(source))


@dataclass(frozen=True)
class PhysicsAssetSpec:
    """Reusable definition of one complete physical asset."""

    schema_version: str
    asset_id: str
    name: str
    materials: tuple[PhysicsMaterialSpec, ...]
    bodies: tuple[RigidBodySpec, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _non_empty_string(
                self.schema_version,
                field_name="schema_version",
                error_type=InvalidPhysicsAssetError,
            ),
        )
        if self.schema_version != CURRENT_PHYSICS_ASSET_SCHEMA_VERSION:
            raise InvalidPhysicsAssetError(
                "schema_version must be '1.0' for PhysicsAssetSpec; "
                f"actual value={self.schema_version!r}"
            )
        object.__setattr__(
            self,
            "asset_id",
            _non_empty_string(
                self.asset_id,
                field_name="asset_id",
                error_type=InvalidPhysicsAssetError,
            ),
        )
        object.__setattr__(
            self,
            "name",
            _non_empty_string(
                self.name,
                field_name="name",
                error_type=InvalidPhysicsAssetError,
            ),
        )
        materials = tuple(self.materials)
        bodies = tuple(self.bodies)
        for material in materials:
            validate_material(material)
        for body in bodies:
            validate_rigid_body(body)
        object.__setattr__(self, "materials", materials)
        object.__setattr__(self, "bodies", bodies)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        validate_physics_asset(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the asset to a JSON-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "asset_id": self.asset_id,
            "name": self.name,
            "materials": [material.to_dict() for material in self.materials],
            "bodies": [body.to_dict() for body in self.bodies],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhysicsAssetSpec":
        """Deserialize an asset from a dictionary."""
        if not isinstance(data, dict):
            raise InvalidPhysicsAssetError(f"asset data must be a dict; actual value={data!r}")
        return cls(
            schema_version=data.get("schema_version"),
            asset_id=data.get("asset_id"),
            name=data.get("name"),
            materials=tuple(
                PhysicsMaterialSpec.from_dict(item) for item in data.get("materials", ())
            ),
            bodies=tuple(RigidBodySpec.from_dict(item) for item in data.get("bodies", ())),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Serialize the asset to stable, readable JSON."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "PhysicsAssetSpec":
        """Deserialize an asset from JSON text."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SerializationError(f"invalid physics asset JSON: {exc.msg} at position {exc.pos}") from exc
        return cls.from_dict(data)


def create_single_body_asset(
    *,
    asset_id: str,
    body: RigidBodySpec,
    name: Optional[str] = None,
    materials: Optional[tuple[PhysicsMaterialSpec, ...]] = None,
    metadata: Optional[dict[str, str]] = None,
) -> PhysicsAssetSpec:
    """Create a reusable asset containing exactly one rigid body."""
    validate_rigid_body(body)
    selected_materials = (DEFAULT_MATERIAL,) if materials is None else tuple(materials)
    return PhysicsAssetSpec(
        schema_version=CURRENT_PHYSICS_ASSET_SCHEMA_VERSION,
        asset_id=asset_id,
        name=name or body.name,
        materials=selected_materials,
        bodies=(body,),
        metadata={} if metadata is None else metadata,
    )
