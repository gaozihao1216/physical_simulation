"""Utilities for explicit physical scale baking.

Physics IR geometry stores final physical dimensions. Rigid body, collider, and
scene instance transforms must not hide physical scale in ``Transform.scale``.
"""

from __future__ import annotations

from physical_simulation.assets.geometry import (
    BoxGeometry,
    CapsuleGeometry,
    ConeGeometry,
    CylinderGeometry,
    EllipsoidGeometry,
    FrustumGeometry,
    GeometrySpec,
    RegularPrismGeometry,
    SphericalCapGeometry,
    SphereGeometry,
    WedgeGeometry,
)
from physical_simulation.assets.transform import Transform
from physical_simulation.validation.asset_validator import _as_float_tuple
from physical_simulation.validation.errors import InvalidGeometryError, ScaleBakingError


UNIT_SCALE = (1.0, 1.0, 1.0)


def is_unit_scale(
    scale: tuple[float, float, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether a scale is equivalent to unit scale within tolerance."""
    values = _as_float_tuple(
        scale,
        field_name="scale",
        length=3,
        strictly_positive=True,
        error_type=ScaleBakingError,
    )
    return all(abs(value - 1.0) <= tolerance for value in values)


def _close(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance


def _scale_error(geometry: GeometrySpec, scale: tuple[float, float, float], reason: str) -> ScaleBakingError:
    return ScaleBakingError(
        f"cannot bake scale into geometry type={type(geometry).__name__}; "
        f"scale={scale!r}; reason={reason}"
    )


def bake_scale_into_geometry(
    geometry: GeometrySpec,
    scale: tuple[float, float, float],
    *,
    tolerance: float = 1e-9,
) -> GeometrySpec:
    """Bake an explicit positive scale into supported analytic geometry.

    Box, wedge, and ellipsoid support arbitrary non-uniform positive scale.
    Axisymmetric shapes require equal X/Y radial scale unless they explicitly
    preserve spherical caps, in which case uniform scale is required.
    """
    sx, sy, sz = _as_float_tuple(
        scale,
        field_name="scale",
        length=3,
        strictly_positive=True,
        error_type=ScaleBakingError,
    )
    if is_unit_scale((sx, sy, sz), tolerance=tolerance):
        return geometry
    if isinstance(geometry, BoxGeometry):
        x, y, z = geometry.size
        return BoxGeometry(size=(x * sx, y * sy, z * sz))
    if isinstance(geometry, WedgeGeometry):
        x, y, z = geometry.size
        return WedgeGeometry(size=(x * sx, y * sy, z * sz))
    if isinstance(geometry, SphereGeometry):
        if not (_close(sx, sy, tolerance) and _close(sx, sz, tolerance)):
            raise _scale_error(
                geometry,
                (sx, sy, sz),
                "SphereGeometry only supports uniform scale; non-uniform scale would create an ellipsoid",
            )
        return SphereGeometry(radius=geometry.radius * sx)
    if isinstance(geometry, CylinderGeometry):
        if not _close(sx, sy, tolerance):
            raise _scale_error(
                geometry,
                (sx, sy, sz),
                "CylinderGeometry requires equal X/Y scale; non-uniform radial scale would create an elliptical cylinder",
            )
        return CylinderGeometry(radius=geometry.radius * sx, height=geometry.height * sz)
    if isinstance(geometry, CapsuleGeometry):
        if not (_close(sx, sy, tolerance) and _close(sx, sz, tolerance)):
            raise _scale_error(
                geometry,
                (sx, sy, sz),
                "CapsuleGeometry only supports uniform scale in Phase 1.5 to preserve spherical caps exactly",
            )
        return CapsuleGeometry(radius=geometry.radius * sx, length=geometry.length * sx)
    if isinstance(geometry, ConeGeometry):
        if not _close(sx, sy, tolerance):
            raise _scale_error(
                geometry,
                (sx, sy, sz),
                "ConeGeometry requires equal X/Y scale; non-uniform radial scale would create an elliptical cone",
            )
        return ConeGeometry(radius=geometry.radius * sx, height=geometry.height * sz)
    if isinstance(geometry, FrustumGeometry):
        if not _close(sx, sy, tolerance):
            raise _scale_error(
                geometry,
                (sx, sy, sz),
                "FrustumGeometry requires equal X/Y scale; non-uniform radial scale would create an elliptical frustum",
            )
        return FrustumGeometry(
            bottom_radius=geometry.bottom_radius * sx,
            top_radius=geometry.top_radius * sx,
            height=geometry.height * sz,
        )
    if isinstance(geometry, EllipsoidGeometry):
        rx, ry, rz = geometry.radii
        return EllipsoidGeometry(radii=(rx * sx, ry * sy, rz * sz))
    if isinstance(geometry, SphericalCapGeometry):
        if not (_close(sx, sy, tolerance) and _close(sx, sz, tolerance)):
            raise _scale_error(
                geometry,
                (sx, sy, sz),
                "SphericalCapGeometry only supports uniform scale; non-uniform scale would create an ellipsoidal cap",
            )
        return SphericalCapGeometry(radius=geometry.radius * sx, height=geometry.height * sx)
    if isinstance(geometry, RegularPrismGeometry):
        if not _close(sx, sy, tolerance):
            raise _scale_error(
                geometry,
                (sx, sy, sz),
                "RegularPrismGeometry requires equal X/Y scale to preserve a regular polygon base",
            )
        return RegularPrismGeometry(
            sides=geometry.sides,
            radius=geometry.radius * sx,
            height=geometry.height * sz,
        )
    raise InvalidGeometryError(
        "geometry must be a supported analytic GeometrySpec; "
        f"actual value={geometry!r}"
    )


def bake_transform_scale(
    geometry: GeometrySpec,
    transform: Transform,
) -> tuple[GeometrySpec, Transform]:
    """Bake transform scale into geometry and return a unit-scale transform."""
    if not isinstance(transform, Transform):
        raise ScaleBakingError(f"transform must be Transform; actual value={transform!r}")
    baked_geometry = bake_scale_into_geometry(geometry, transform.scale)
    baked_transform = Transform(
        position=transform.position,
        rotation=transform.rotation,
        scale=UNIT_SCALE,
    )
    return baked_geometry, baked_transform
