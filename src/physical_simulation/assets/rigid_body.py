"""Rigid body specification for backend-independent Physics IR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Optional

from physical_simulation.assets.collider import ColliderSpec
from physical_simulation.assets.mass_properties import MassProperties
from physical_simulation.assets.transform import Transform
from physical_simulation.assets.visual import VisualSpec
from physical_simulation.validation.asset_validator import (
    _non_empty_string,
    validate_collider,
    validate_mass_properties,
    validate_transform,
    validate_visual,
)
from physical_simulation.validation.errors import InvalidRigidBodyError, SerializationError

BodyType = Literal["static", "kinematic", "dynamic"]


@dataclass(frozen=True)
class RigidBodySpec:
    """Single rigid body with independent visual and collision specifications."""

    body_id: str
    name: str
    body_type: BodyType
    transform: Transform
    visuals: tuple[VisualSpec, ...]
    colliders: tuple[ColliderSpec, ...]
    mass_properties: Optional[MassProperties] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "body_id",
            _non_empty_string(
                self.body_id,
                field_name="body_id",
                error_type=InvalidRigidBodyError,
            ),
        )
        object.__setattr__(
            self,
            "name",
            _non_empty_string(
                self.name,
                field_name="name",
                error_type=InvalidRigidBodyError,
            ),
        )
        if self.body_type not in ("static", "kinematic", "dynamic"):
            raise InvalidRigidBodyError(
                "body_type must be 'static', 'kinematic', or 'dynamic'; "
                f"actual value={self.body_type!r}"
            )
        validate_transform(self.transform)
        visuals = tuple(self.visuals)
        colliders = tuple(self.colliders)
        for visual in visuals:
            validate_visual(visual)
        for collider in colliders:
            validate_collider(collider)
        object.__setattr__(self, "visuals", visuals)
        object.__setattr__(self, "colliders", colliders)
        visual_ids = [visual.visual_id for visual in visuals]
        collider_ids = [collider.collider_id for collider in colliders]
        if len(visual_ids) != len(set(visual_ids)):
            raise InvalidRigidBodyError(
                f"visual IDs must be unique within body_id={self.body_id!r}; actual IDs={visual_ids!r}"
            )
        if len(collider_ids) != len(set(collider_ids)):
            raise InvalidRigidBodyError(
                f"collider IDs must be unique within body_id={self.body_id!r}; actual IDs={collider_ids!r}"
            )
        if self.mass_properties is not None:
            validate_mass_properties(self.mass_properties)
        if self.body_type == "dynamic":
            if self.mass_properties is None:
                raise InvalidRigidBodyError(
                    f"mass_properties is required for dynamic body; body_id={self.body_id!r}"
                )
            if not any(collider.enabled for collider in colliders):
                raise InvalidRigidBodyError(
                    f"dynamic body must have at least one enabled collider; body_id={self.body_id!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the rigid body to a JSON-compatible dictionary."""
        return {
            "body_id": self.body_id,
            "name": self.name,
            "body_type": self.body_type,
            "transform": self.transform.to_dict(),
            "visuals": [visual.to_dict() for visual in self.visuals],
            "colliders": [collider.to_dict() for collider in self.colliders],
            "mass_properties": None
            if self.mass_properties is None
            else self.mass_properties.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RigidBodySpec":
        """Deserialize a rigid body from a dictionary."""
        if not isinstance(data, dict):
            raise InvalidRigidBodyError(f"rigid body data must be a dict; actual value={data!r}")
        mass_data = data.get("mass_properties")
        return cls(
            body_id=data.get("body_id"),
            name=data.get("name"),
            body_type=data.get("body_type"),
            transform=Transform.from_dict(data.get("transform", {})),
            visuals=tuple(VisualSpec.from_dict(item) for item in data.get("visuals", ())),
            colliders=tuple(ColliderSpec.from_dict(item) for item in data.get("colliders", ())),
            mass_properties=None if mass_data is None else MassProperties.from_dict(mass_data),
        )

    def to_json(self) -> str:
        """Serialize the rigid body to stable, readable JSON."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RigidBodySpec":
        """Deserialize a rigid body from JSON text."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SerializationError(f"invalid rigid body JSON: {exc.msg} at position {exc.pos}") from exc
        return cls.from_dict(data)
