"""Validation functions for backend-independent physics asset data."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

from physical_simulation.validation.errors import (
    InvalidGeometryError,
    InvalidMassPropertiesError,
    InvalidRigidBodyError,
    PhysicsValidationError,
)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _as_float_tuple(
    value: Any,
    *,
    field_name: str,
    length: int,
    positive: bool = False,
    strictly_positive: bool = False,
    error_type: type[PhysicsValidationError] = PhysicsValidationError,
) -> tuple[float, ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise error_type(
            f"{field_name} must be a sequence of {length} finite numbers; actual value={value!r}"
        )
    values = tuple(value)
    if len(values) != length:
        raise error_type(
            f"{field_name} must contain {length} values; actual value={value!r}"
        )
    result = []
    for item in values:
        if not _is_finite_number(item):
            raise error_type(
                f"{field_name} must contain finite numeric values; actual value={value!r}"
            )
        number = float(item)
        if positive and number < 0.0:
            raise error_type(
                f"{field_name} must be >= 0; actual value={value!r}"
            )
        if strictly_positive and number <= 0.0:
            raise error_type(
                f"{field_name} must be > 0; actual value={value!r}"
            )
        result.append(number)
    return tuple(result)


def _finite_float(
    value: Any,
    *,
    field_name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    strict_minimum: bool = False,
    error_type: type[PhysicsValidationError] = PhysicsValidationError,
) -> float:
    if not _is_finite_number(value):
        raise error_type(f"{field_name} must be a finite number; actual value={value!r}")
    number = float(value)
    if minimum is not None:
        if strict_minimum and number <= minimum:
            raise error_type(f"{field_name} must be > {minimum}; actual value={value!r}")
        if not strict_minimum and number < minimum:
            raise error_type(f"{field_name} must be >= {minimum}; actual value={value!r}")
    if maximum is not None and number > maximum:
        raise error_type(f"{field_name} must be <= {maximum}; actual value={value!r}")
    return number


def _non_empty_string(value: Any, *, field_name: str, error_type: type[PhysicsValidationError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{field_name} must be a non-empty string; actual value={value!r}")
    return value


def validate_transform(transform: Any) -> None:
    """Validate a Transform instance."""
    from physical_simulation.assets.transform import Transform

    if not isinstance(transform, Transform):
        raise PhysicsValidationError(
            f"transform must be Transform; actual type={type(transform).__name__}, value={transform!r}"
        )


def validate_geometry(geometry: Any) -> None:
    """Validate a geometry specification instance."""
    from physical_simulation.assets.geometry import (
        BoxGeometry,
        CapsuleGeometry,
        CylinderGeometry,
        SphereGeometry,
    )

    if not isinstance(geometry, (BoxGeometry, SphereGeometry, CylinderGeometry, CapsuleGeometry)):
        raise InvalidGeometryError(
            "geometry must be BoxGeometry, SphereGeometry, CylinderGeometry, or CapsuleGeometry; "
            f"actual type={type(geometry).__name__}, value={geometry!r}"
        )


def validate_material(material: Any) -> None:
    """Validate a PhysicsMaterialSpec instance."""
    from physical_simulation.assets.material import PhysicsMaterialSpec

    if not isinstance(material, PhysicsMaterialSpec):
        raise PhysicsValidationError(
            f"material must be PhysicsMaterialSpec; actual type={type(material).__name__}, value={material!r}"
        )


def validate_mass_properties(mass_properties: Any) -> None:
    """Validate a MassProperties instance."""
    from physical_simulation.assets.mass_properties import MassProperties

    if not isinstance(mass_properties, MassProperties):
        raise InvalidMassPropertiesError(
            "mass_properties must be MassProperties; "
            f"actual type={type(mass_properties).__name__}, value={mass_properties!r}"
        )


def validate_visual(visual: Any) -> None:
    """Validate a VisualSpec instance."""
    from physical_simulation.assets.visual import VisualSpec

    if not isinstance(visual, VisualSpec):
        raise PhysicsValidationError(
            f"visual must be VisualSpec; actual type={type(visual).__name__}, value={visual!r}"
        )


def validate_collider(collider: Any) -> None:
    """Validate a ColliderSpec instance."""
    from physical_simulation.assets.collider import ColliderSpec

    if not isinstance(collider, ColliderSpec):
        raise PhysicsValidationError(
            f"collider must be ColliderSpec; actual type={type(collider).__name__}, value={collider!r}"
        )


def validate_rigid_body(body: Any) -> None:
    """Validate a RigidBodySpec instance."""
    from physical_simulation.assets.rigid_body import RigidBodySpec

    if not isinstance(body, RigidBodySpec):
        raise InvalidRigidBodyError(
            f"body must be RigidBodySpec; actual type={type(body).__name__}, value={body!r}"
        )


__all__ = [
    "_as_float_tuple",
    "_finite_float",
    "_non_empty_string",
    "validate_transform",
    "validate_geometry",
    "validate_material",
    "validate_mass_properties",
    "validate_visual",
    "validate_collider",
    "validate_rigid_body",
]
