"""Basic analytic geometry specifications for Physics IR assets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Union

from physical_simulation.validation.asset_validator import _as_float_tuple, _finite_float
from physical_simulation.validation.errors import InvalidGeometryError


@dataclass(frozen=True)
class BoxGeometry:
    """Box geometry whose size stores full extents along X, Y, and Z."""

    size: tuple[float, float, float]
    shape_type: str = "box"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "size",
            _as_float_tuple(
                self.size,
                field_name="size",
                length=3,
                strictly_positive=True,
                error_type=InvalidGeometryError,
            ),
        )

    def volume(self) -> float:
        """Return volume in cubic meters."""
        x, y, z = self.size
        return x * y * z

    def to_dict(self) -> dict[str, Any]:
        """Serialize the geometry to a JSON-compatible dictionary."""
        return {"shape_type": self.shape_type, "size": list(self.size)}


@dataclass(frozen=True)
class SphereGeometry:
    """Sphere geometry."""

    radius: float
    shape_type: str = "sphere"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "radius",
            _finite_float(
                self.radius,
                field_name="radius",
                minimum=0.0,
                strict_minimum=True,
                error_type=InvalidGeometryError,
            ),
        )

    def volume(self) -> float:
        """Return volume in cubic meters."""
        return 4.0 / 3.0 * math.pi * self.radius**3

    def to_dict(self) -> dict[str, Any]:
        """Serialize the geometry to a JSON-compatible dictionary."""
        return {"shape_type": self.shape_type, "radius": self.radius}


@dataclass(frozen=True)
class CylinderGeometry:
    """Cylinder geometry aligned to the local Z axis."""

    radius: float
    height: float
    shape_type: str = "cylinder"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "radius",
            _finite_float(
                self.radius,
                field_name="radius",
                minimum=0.0,
                strict_minimum=True,
                error_type=InvalidGeometryError,
            ),
        )
        object.__setattr__(
            self,
            "height",
            _finite_float(
                self.height,
                field_name="height",
                minimum=0.0,
                strict_minimum=True,
                error_type=InvalidGeometryError,
            ),
        )

    def volume(self) -> float:
        """Return volume in cubic meters."""
        return math.pi * self.radius**2 * self.height

    def to_dict(self) -> dict[str, Any]:
        """Serialize the geometry to a JSON-compatible dictionary."""
        return {"shape_type": self.shape_type, "radius": self.radius, "height": self.height}


@dataclass(frozen=True)
class CapsuleGeometry:
    """Capsule geometry aligned to the local Z axis.

    ``length`` is the cylindrical section between the two hemispheres.
    """

    radius: float
    length: float
    shape_type: str = "capsule"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "radius",
            _finite_float(
                self.radius,
                field_name="radius",
                minimum=0.0,
                strict_minimum=True,
                error_type=InvalidGeometryError,
            ),
        )
        object.__setattr__(
            self,
            "length",
            _finite_float(
                self.length,
                field_name="length",
                minimum=0.0,
                strict_minimum=True,
                error_type=InvalidGeometryError,
            ),
        )

    def volume(self) -> float:
        """Return volume in cubic meters."""
        return math.pi * self.radius**2 * self.length + 4.0 / 3.0 * math.pi * self.radius**3

    def to_dict(self) -> dict[str, Any]:
        """Serialize the geometry to a JSON-compatible dictionary."""
        return {"shape_type": self.shape_type, "radius": self.radius, "length": self.length}


GeometrySpec = Union[BoxGeometry, SphereGeometry, CylinderGeometry, CapsuleGeometry]


def geometry_from_dict(data: dict[str, Any]) -> GeometrySpec:
    """Deserialize one of the supported analytic geometry specifications."""
    if not isinstance(data, dict):
        raise InvalidGeometryError(f"geometry data must be a dict; actual value={data!r}")
    shape_type = data.get("shape_type")
    if shape_type == "box":
        return BoxGeometry(size=tuple(data.get("size", ())))
    if shape_type == "sphere":
        return SphereGeometry(radius=data.get("radius"))
    if shape_type == "cylinder":
        return CylinderGeometry(radius=data.get("radius"), height=data.get("height"))
    if shape_type == "capsule":
        return CapsuleGeometry(radius=data.get("radius"), length=data.get("length"))
    raise InvalidGeometryError(
        "shape_type must be one of 'box', 'sphere', 'cylinder', or 'capsule'; "
        f"actual value={shape_type!r}"
    )
