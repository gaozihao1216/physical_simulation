"""Immutable physics scene input specification."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

from physical_simulation.assets.physics_asset import create_single_body_asset
from physical_simulation.assets.rigid_body import RigidBodySpec
from physical_simulation.assets.transform import Transform
from physical_simulation.scene.asset_instance import AssetInstanceSpec
from physical_simulation.validation.asset_validator import (
    _as_float_tuple,
    _finite_float,
    _non_empty_string,
    validate_physics_scene,
)
from physical_simulation.validation.errors import InvalidPhysicsSceneError, SerializationError

CURRENT_PHYSICS_SCENE_SCHEMA_VERSION = "1.0"


def _freeze_metadata(metadata: Mapping[str, str] | None) -> Mapping[str, str]:
    source = {} if metadata is None else dict(metadata)
    for key, value in source.items():
        if not isinstance(key, str) or not key.strip():
            raise InvalidPhysicsSceneError(
                f"metadata keys must be non-empty strings; actual key={key!r}"
            )
        if not isinstance(value, str):
            raise InvalidPhysicsSceneError(
                f"metadata values must be strings; actual key={key!r}, value={value!r}"
            )
    return MappingProxyType(dict(source))


@dataclass(frozen=True)
class PhysicsSceneSpec:
    """Backend-independent immutable input scene specification."""

    schema_version: str
    scene_id: str
    gravity: tuple[float, float, float]
    timestep: float
    instances: tuple[AssetInstanceSpec, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _non_empty_string(
                self.schema_version,
                field_name="schema_version",
                error_type=InvalidPhysicsSceneError,
            ),
        )
        if self.schema_version != CURRENT_PHYSICS_SCENE_SCHEMA_VERSION:
            raise InvalidPhysicsSceneError(
                "schema_version must be '1.0' for PhysicsSceneSpec; "
                f"actual value={self.schema_version!r}"
            )
        object.__setattr__(
            self,
            "scene_id",
            _non_empty_string(
                self.scene_id,
                field_name="scene_id",
                error_type=InvalidPhysicsSceneError,
            ),
        )
        object.__setattr__(
            self,
            "gravity",
            _as_float_tuple(
                self.gravity,
                field_name="gravity",
                length=3,
                error_type=InvalidPhysicsSceneError,
            ),
        )
        object.__setattr__(
            self,
            "timestep",
            _finite_float(
                self.timestep,
                field_name="timestep",
                minimum=0.0,
                strict_minimum=True,
                error_type=InvalidPhysicsSceneError,
            ),
        )
        object.__setattr__(self, "instances", tuple(self.instances))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        validate_physics_scene(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the scene to a JSON-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "gravity": list(self.gravity),
            "timestep": self.timestep,
            "instances": [instance.to_dict() for instance in self.instances],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhysicsSceneSpec":
        """Deserialize a scene from a dictionary."""
        if not isinstance(data, dict):
            raise InvalidPhysicsSceneError(f"scene data must be a dict; actual value={data!r}")
        return cls(
            schema_version=data.get("schema_version"),
            scene_id=data.get("scene_id"),
            gravity=tuple(data.get("gravity", ())),
            timestep=data.get("timestep"),
            instances=tuple(
                AssetInstanceSpec.from_dict(item) for item in data.get("instances", ())
            ),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Serialize the scene to stable, readable JSON."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "PhysicsSceneSpec":
        """Deserialize a scene from JSON text."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SerializationError(f"invalid physics scene JSON: {exc.msg} at position {exc.pos}") from exc
        return cls.from_dict(data)


def create_scene(
    *,
    scene_id: str,
    instances: tuple[AssetInstanceSpec, ...],
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    timestep: float = 1.0 / 240.0,
    metadata: Optional[dict[str, str]] = None,
) -> PhysicsSceneSpec:
    """Create a physics scene specification."""
    return PhysicsSceneSpec(
        schema_version=CURRENT_PHYSICS_SCENE_SCHEMA_VERSION,
        scene_id=scene_id,
        gravity=gravity,
        timestep=timestep,
        instances=instances,
        metadata={} if metadata is None else metadata,
    )


def create_body_instance(
    *,
    instance_id: str,
    body: RigidBodySpec,
    transform: Optional[Transform] = None,
    fixed_base: bool = False,
    asset_id: Optional[str] = None,
) -> AssetInstanceSpec:
    """Wrap a single rigid body in an asset and place it as a scene instance."""
    asset = create_single_body_asset(
        asset_id=asset_id or f"{body.body_id}_asset",
        body=body,
    )
    return AssetInstanceSpec(
        instance_id=instance_id,
        asset=asset,
        transform=Transform.identity() if transform is None else transform,
        fixed_base=fixed_base,
    )
