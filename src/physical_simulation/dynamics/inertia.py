"""Analytic and approximate diagonal inertia calculations."""

from __future__ import annotations

import math

from physical_simulation.assets.geometry import (
    BoxGeometry,
    CapsuleGeometry,
    CylinderGeometry,
    GeometrySpec,
    SphereGeometry,
)
from physical_simulation.validation.asset_validator import _as_float_tuple, _finite_float
from physical_simulation.validation.errors import InvalidGeometryError, InvalidMassPropertiesError


def _validate_mass(mass: float) -> float:
    return _finite_float(
        mass,
        field_name="mass",
        minimum=0.0,
        strict_minimum=True,
        error_type=InvalidMassPropertiesError,
    )


def compute_box_inertia(mass: float, size: tuple[float, float, float]) -> tuple[float, float, float]:
    """Compute diagonal inertia for a box with full extents."""
    m = _validate_mass(mass)
    x, y, z = _as_float_tuple(
        size,
        field_name="size",
        length=3,
        strictly_positive=True,
        error_type=InvalidGeometryError,
    )
    return (
        m / 12.0 * (y**2 + z**2),
        m / 12.0 * (x**2 + z**2),
        m / 12.0 * (x**2 + y**2),
    )


def compute_sphere_inertia(mass: float, radius: float) -> tuple[float, float, float]:
    """Compute diagonal inertia for a solid sphere."""
    m = _validate_mass(mass)
    r = _finite_float(
        radius,
        field_name="radius",
        minimum=0.0,
        strict_minimum=True,
        error_type=InvalidGeometryError,
    )
    inertia = 2.0 / 5.0 * m * r**2
    return (inertia, inertia, inertia)


def compute_cylinder_inertia(mass: float, radius: float, height: float) -> tuple[float, float, float]:
    """Compute diagonal inertia for a solid cylinder aligned to the local Z axis."""
    m = _validate_mass(mass)
    r = _finite_float(
        radius,
        field_name="radius",
        minimum=0.0,
        strict_minimum=True,
        error_type=InvalidGeometryError,
    )
    h = _finite_float(
        height,
        field_name="height",
        minimum=0.0,
        strict_minimum=True,
        error_type=InvalidGeometryError,
    )
    transverse = m / 12.0 * (3.0 * r**2 + h**2)
    axial = 0.5 * m * r**2
    return (transverse, transverse, axial)


def compute_capsule_inertia(mass: float, radius: float, length: float) -> tuple[float, float, float]:
    """Approximate diagonal inertia for a solid capsule aligned to local Z.

    The capsule is split into a cylinder and two hemispheres. Mass is distributed by
    volume. The two hemispheres are approximated as half-sphere masses whose centers
    lie at ``length / 2 + 3 * radius / 8`` from the capsule origin. The transverse
    inertia uses the parallel-axis theorem; axial inertia is unaffected by the
    displacement along Z. This is a clear first-phase approximation, not a full
    closed-form inertia tensor for a capsule.
    """
    m = _validate_mass(mass)
    r = _finite_float(
        radius,
        field_name="radius",
        minimum=0.0,
        strict_minimum=True,
        error_type=InvalidGeometryError,
    )
    capsule_length = _finite_float(
        length,
        field_name="length",
        minimum=0.0,
        strict_minimum=True,
        error_type=InvalidGeometryError,
    )
    cylinder_volume = math.pi * r**2 * capsule_length
    sphere_volume = 4.0 / 3.0 * math.pi * r**3
    total_volume = cylinder_volume + sphere_volume
    cylinder_mass = m * cylinder_volume / total_volume
    sphere_mass = m - cylinder_mass
    cylinder_ix, _, cylinder_iz = compute_cylinder_inertia(cylinder_mass, r, capsule_length)

    hemisphere_center_offset = capsule_length / 2.0 + 3.0 * r / 8.0
    sphere_axial = 2.0 / 5.0 * sphere_mass * r**2
    sphere_transverse = sphere_axial + sphere_mass * hemisphere_center_offset**2

    transverse = cylinder_ix + sphere_transverse
    axial = cylinder_iz + sphere_axial
    return (transverse, transverse, axial)


def compute_inertia(geometry: GeometrySpec, mass: float) -> tuple[float, float, float]:
    """Compute diagonal inertia for a supported geometry specification."""
    if isinstance(geometry, BoxGeometry):
        return compute_box_inertia(mass, geometry.size)
    if isinstance(geometry, SphereGeometry):
        return compute_sphere_inertia(mass, geometry.radius)
    if isinstance(geometry, CylinderGeometry):
        return compute_cylinder_inertia(mass, geometry.radius, geometry.height)
    if isinstance(geometry, CapsuleGeometry):
        return compute_capsule_inertia(mass, geometry.radius, geometry.length)
    raise InvalidGeometryError(
        "geometry must be BoxGeometry, SphereGeometry, CylinderGeometry, or CapsuleGeometry; "
        f"actual value={geometry!r}"
    )


def compute_mass_from_density(geometry: GeometrySpec, density: float) -> float:
    """Compute mass from geometry volume and density."""
    rho = _finite_float(
        density,
        field_name="density",
        minimum=0.0,
        strict_minimum=True,
        error_type=InvalidMassPropertiesError,
    )
    if not hasattr(geometry, "volume"):
        raise InvalidGeometryError(f"geometry must provide volume(); actual value={geometry!r}")
    return geometry.volume() * rho
