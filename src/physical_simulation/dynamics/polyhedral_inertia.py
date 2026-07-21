"""Full tensor mass properties for polyhedral and frustum geometry."""

from __future__ import annotations

from physical_simulation.assets.geometry import FrustumGeometry, GeometrySpec, RegularPrismGeometry, WedgeGeometry
from physical_simulation.collision.convex_mesh import ConvexMeshSpec, regular_prism_to_convex_mesh, wedge_to_convex_mesh
from physical_simulation.dynamics.compound_inertia import (
    CompoundMassProperties,
    Matrix3,
    Vector3,
    diagonalize_symmetric_tensor,
)
from physical_simulation.validation.asset_validator import _finite_float
from physical_simulation.validation.errors import InvalidGeometryError, InvalidMassPropertiesError


def compute_wedge_mass_properties(geometry: WedgeGeometry, mass: float) -> CompoundMassProperties:
    """Compute full tensor mass properties for a solid wedge/ramp."""
    return compute_polyhedral_mass_properties(wedge_to_convex_mesh(geometry), mass)


def compute_regular_prism_mass_properties(
    geometry: RegularPrismGeometry,
    mass: float,
) -> CompoundMassProperties:
    """Compute full tensor mass properties for a solid regular polygon prism."""
    return compute_polyhedral_mass_properties(regular_prism_to_convex_mesh(geometry), mass)


def compute_frustum_mass_properties(geometry: FrustumGeometry, mass: float) -> CompoundMassProperties:
    """Compute exact full tensor mass properties for a solid circular frustum."""
    m = _validate_mass(mass)
    r1 = geometry.bottom_radius
    r2 = geometry.top_radius
    h = geometry.height
    s2 = r1**2 + r1 * r2 + r2**2
    a0 = h * s2 / 3.0
    a1 = h**2 * (r1**2 / 2.0 + 2.0 * r1 * (r2 - r1) / 3.0 + (r2 - r1) ** 2 / 4.0)
    a2 = h**3 * (r1**2 / 3.0 + r1 * (r2 - r1) / 2.0 + (r2 - r1) ** 2 / 5.0)
    z_from_bottom = a1 / a0
    center_of_mass = (0.0, 0.0, z_from_bottom - h / 2.0)

    radial_fourth = h * (r1**4 + r1**3 * r2 + r1**2 * r2**2 + r1 * r2**3 + r2**4) / 5.0
    variance_z_integral = a2 - a1 * a1 / a0
    transverse = m / a0 * (0.25 * radial_fourth + variance_z_integral)
    axial = m / a0 * (0.5 * radial_fourth)
    tensor = ((transverse, 0.0, 0.0), (0.0, transverse, 0.0), (0.0, 0.0, axial))
    principal_inertia, principal_axes = diagonalize_symmetric_tensor(tensor)
    return CompoundMassProperties(
        mass=m,
        center_of_mass=center_of_mass,
        inertia_tensor=tensor,
        principal_inertia=principal_inertia,
        principal_axes=principal_axes,
    )


def compute_polyhedral_mass_properties(mesh: ConvexMeshSpec, mass: float) -> CompoundMassProperties:
    """Compute full tensor mass properties from a closed triangular polyhedron.

    The tensor is exact for the supplied polyhedron. Faces may use either winding;
    each triangle is oriented consistently with the origin tetrahedron during
    accumulation.
    """
    m = _validate_mass(mass)
    total_volume = 0.0
    first_moment = [0.0, 0.0, 0.0]
    second_moment = [[0.0, 0.0, 0.0] for _ in range(3)]

    for face in mesh.faces:
        a, b, c = (mesh.vertices[index] for index in face)
        signed_volume = _determinant3(a, b, c) / 6.0
        if abs(signed_volume) <= 1.0e-15:
            continue
        if signed_volume < 0.0:
            b, c = c, b
            signed_volume = -signed_volume
        total_volume += signed_volume
        centroid = tuple((a[axis] + b[axis] + c[axis]) / 4.0 for axis in range(3))
        for axis in range(3):
            first_moment[axis] += signed_volume * centroid[axis]

        vectors = (a, b, c)
        summed = tuple(a[axis] + b[axis] + c[axis] for axis in range(3))
        for row in range(3):
            for col in range(3):
                vertex_outer = sum(vector[row] * vector[col] for vector in vectors)
                second_moment[row][col] += signed_volume / 20.0 * (
                    summed[row] * summed[col] + vertex_outer
                )

    if total_volume <= 0.0:
        raise InvalidGeometryError("polyhedral mesh volume must be > 0")
    density = m / total_volume
    center_of_mass = tuple(moment / total_volume for moment in first_moment)
    second_moment_mass = tuple(
        tuple(density * second_moment[row][col] for col in range(3))
        for row in range(3)
    )
    inertia_about_origin = _second_moment_to_inertia(second_moment_mass)
    inertia_tensor = _translate_origin_inertia_to_com(inertia_about_origin, m, center_of_mass)
    inertia_tensor = _symmetrize(inertia_tensor)
    principal_inertia, principal_axes = diagonalize_symmetric_tensor(inertia_tensor)
    return CompoundMassProperties(
        mass=m,
        center_of_mass=center_of_mass,  # type: ignore[arg-type]
        inertia_tensor=inertia_tensor,
        principal_inertia=principal_inertia,
        principal_axes=principal_axes,
    )


def compute_full_geometry_mass_properties(geometry: GeometrySpec, mass: float) -> CompoundMassProperties:
    """Compute full tensor mass properties for supported non-primitive geometries."""
    if isinstance(geometry, WedgeGeometry):
        return compute_wedge_mass_properties(geometry, mass)
    if isinstance(geometry, FrustumGeometry):
        return compute_frustum_mass_properties(geometry, mass)
    if isinstance(geometry, RegularPrismGeometry):
        return compute_regular_prism_mass_properties(geometry, mass)
    raise InvalidGeometryError(
        "full geometry mass properties are implemented for WedgeGeometry, FrustumGeometry, "
        f"and RegularPrismGeometry; actual value={geometry!r}"
    )


def _validate_mass(mass: float) -> float:
    return _finite_float(
        mass,
        field_name="mass",
        minimum=0.0,
        strict_minimum=True,
        error_type=InvalidMassPropertiesError,
    )


def _determinant3(a: Vector3, b: Vector3, c: Vector3) -> float:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _second_moment_to_inertia(moment: Matrix3) -> Matrix3:
    trace = moment[0][0] + moment[1][1] + moment[2][2]
    return tuple(
        tuple((trace if row == col else 0.0) - moment[row][col] for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _translate_origin_inertia_to_com(tensor: Matrix3, mass: float, center_of_mass: Vector3) -> Matrix3:
    cx, cy, cz = center_of_mass
    distance_squared = cx * cx + cy * cy + cz * cz
    shift = (
        (mass * (distance_squared - cx * cx), -mass * cx * cy, -mass * cx * cz),
        (-mass * cy * cx, mass * (distance_squared - cy * cy), -mass * cy * cz),
        (-mass * cz * cx, -mass * cz * cy, mass * (distance_squared - cz * cz)),
    )
    return tuple(
        tuple(tensor[row][col] - shift[row][col] for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _symmetrize(matrix: Matrix3) -> Matrix3:
    return tuple(
        tuple(0.5 * (matrix[row][col] + matrix[col][row]) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]
