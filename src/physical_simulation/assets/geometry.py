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


@dataclass(frozen=True)
class WedgeGeometry:
    """Right triangular prism / ramp geometry with full extents along X, Y, and Z."""

    size: tuple[float, float, float]
    shape_type: str = "wedge"

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
        return 0.5 * x * y * z

    def to_dict(self) -> dict[str, Any]:
        """Serialize the geometry to a JSON-compatible dictionary."""
        return {"shape_type": self.shape_type, "size": list(self.size)}


@dataclass(frozen=True)
class ConeGeometry:
    """Solid circular cone aligned to the local Z axis."""

    radius: float
    height: float
    shape_type: str = "cone"

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
        return math.pi * self.radius**2 * self.height / 3.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the geometry to a JSON-compatible dictionary."""
        return {"shape_type": self.shape_type, "radius": self.radius, "height": self.height}


@dataclass(frozen=True)
class FrustumGeometry:
    """Solid circular cone frustum aligned to the local Z axis."""

    bottom_radius: float
    top_radius: float
    height: float
    shape_type: str = "frustum"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bottom_radius",
            _finite_float(
                self.bottom_radius,
                field_name="bottom_radius",
                minimum=0.0,
                strict_minimum=True,
                error_type=InvalidGeometryError,
            ),
        )
        object.__setattr__(
            self,
            "top_radius",
            _finite_float(
                self.top_radius,
                field_name="top_radius",
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
        r1 = self.bottom_radius
        r2 = self.top_radius
        return math.pi * self.height * (r1**2 + r1 * r2 + r2**2) / 3.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the geometry to a JSON-compatible dictionary."""
        return {
            "shape_type": self.shape_type,
            "bottom_radius": self.bottom_radius,
            "top_radius": self.top_radius,
            "height": self.height,
        }


@dataclass(frozen=True)
class EllipsoidGeometry:
    """Solid ellipsoid with radii along local X, Y, and Z axes."""

    radii: tuple[float, float, float]
    shape_type: str = "ellipsoid"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "radii",
            _as_float_tuple(
                self.radii,
                field_name="radii",
                length=3,
                strictly_positive=True,
                error_type=InvalidGeometryError,
            ),
        )

    def volume(self) -> float:
        """Return volume in cubic meters."""
        rx, ry, rz = self.radii
        return 4.0 / 3.0 * math.pi * rx * ry * rz

    def to_dict(self) -> dict[str, Any]:
        """Serialize the geometry to a JSON-compatible dictionary."""
        return {"shape_type": self.shape_type, "radii": list(self.radii)}


@dataclass(frozen=True)
class SphericalCapGeometry:
    """Solid spherical cap cut from a sphere of ``radius`` with cap ``height``."""

    radius: float
    height: float
    shape_type: str = "spherical_cap"

    def __post_init__(self) -> None:
        radius = _finite_float(
            self.radius,
            field_name="radius",
            minimum=0.0,
            strict_minimum=True,
            error_type=InvalidGeometryError,
        )
        height = _finite_float(
            self.height,
            field_name="height",
            minimum=0.0,
            strict_minimum=True,
            error_type=InvalidGeometryError,
        )
        if height > 2.0 * radius:
            raise InvalidGeometryError(
                f"height must be <= 2 * radius for SphericalCapGeometry; radius={radius!r}, height={height!r}"
            )
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "height", height)

    def volume(self) -> float:
        """Return volume in cubic meters."""
        h = self.height
        return math.pi * h**2 * (3.0 * self.radius - h) / 3.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the geometry to a JSON-compatible dictionary."""
        return {"shape_type": self.shape_type, "radius": self.radius, "height": self.height}


@dataclass(frozen=True)
class RegularPrismGeometry:
    """Regular polygon prism aligned to local Z, parameterized by circumradius."""

    sides: int
    radius: float
    height: float
    shape_type: str = "regular_prism"

    def __post_init__(self) -> None:
        if not isinstance(self.sides, int) or isinstance(self.sides, bool) or self.sides < 3:
            raise InvalidGeometryError(f"sides must be an integer >= 3; actual value={self.sides!r}")
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
        base_area = 0.5 * self.sides * self.radius**2 * math.sin(2.0 * math.pi / self.sides)
        return base_area * self.height

    def to_dict(self) -> dict[str, Any]:
        """Serialize the geometry to a JSON-compatible dictionary."""
        return {
            "shape_type": self.shape_type,
            "sides": self.sides,
            "radius": self.radius,
            "height": self.height,
        }


GeometrySpec = Union[
    BoxGeometry,
    SphereGeometry,
    CylinderGeometry,
    CapsuleGeometry,
    WedgeGeometry,
    ConeGeometry,
    FrustumGeometry,
    EllipsoidGeometry,
    SphericalCapGeometry,
    RegularPrismGeometry,
]


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
    if shape_type == "wedge":
        return WedgeGeometry(size=tuple(data.get("size", ())))
    if shape_type == "cone":
        return ConeGeometry(radius=data.get("radius"), height=data.get("height"))
    if shape_type == "frustum":
        return FrustumGeometry(
            bottom_radius=data.get("bottom_radius"),
            top_radius=data.get("top_radius"),
            height=data.get("height"),
        )
    if shape_type == "ellipsoid":
        return EllipsoidGeometry(radii=tuple(data.get("radii", ())))
    if shape_type == "spherical_cap":
        return SphericalCapGeometry(radius=data.get("radius"), height=data.get("height"))
    if shape_type == "regular_prism":
        return RegularPrismGeometry(
            sides=data.get("sides"),
            radius=data.get("radius"),
            height=data.get("height"),
        )
    raise InvalidGeometryError(
        "shape_type must be one of 'box', 'sphere', 'cylinder', 'capsule', 'wedge', "
        "'cone', 'frustum', 'ellipsoid', 'spherical_cap', or 'regular_prism'; "
        f"actual value={shape_type!r}"
    )
