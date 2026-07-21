import math

import pytest

from physical_simulation.assets import BoxGeometry, FrustumGeometry, RegularPrismGeometry, WedgeGeometry
from physical_simulation.collision.convex_mesh import ConvexMeshSpec
from physical_simulation.dynamics import (
    compute_box_inertia,
    compute_frustum_mass_properties,
    compute_full_geometry_mass_properties,
    compute_polyhedral_mass_properties,
    compute_regular_prism_mass_properties,
    compute_wedge_mass_properties,
)
from physical_simulation.dynamics.compound_inertia import Matrix3, diagonal_tensor, multiply_matrix, transpose
from physical_simulation.validation.errors import InvalidGeometryError


def _assert_matrix_close(actual: Matrix3, expected: Matrix3, *, abs: float = 1.0e-12) -> None:
    for row in range(3):
        assert actual[row] == pytest.approx(expected[row], abs=abs)


def _reconstruct_tensor(principal: tuple[float, float, float], axes: Matrix3) -> Matrix3:
    return multiply_matrix(multiply_matrix(axes, diagonal_tensor(principal)), transpose(axes))


def test_polyhedral_mass_properties_match_box_for_cube_mesh() -> None:
    vertices = (
        (-0.5, -1.0, -1.5),
        (0.5, -1.0, -1.5),
        (0.5, 1.0, -1.5),
        (-0.5, 1.0, -1.5),
        (-0.5, -1.0, 1.5),
        (0.5, -1.0, 1.5),
        (0.5, 1.0, 1.5),
        (-0.5, 1.0, 1.5),
    )
    faces = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    )
    mass = 12.0

    result = compute_polyhedral_mass_properties(ConvexMeshSpec(vertices, faces), mass)

    assert result.center_of_mass == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-12)
    _assert_matrix_close(result.inertia_tensor, diagonal_tensor(compute_box_inertia(mass, (1.0, 2.0, 3.0))))


def test_wedge_full_inertia_has_shifted_center_and_product_terms() -> None:
    geometry = WedgeGeometry((2.0, 4.0, 6.0))
    mass = 3.0

    result = compute_wedge_mass_properties(geometry, mass)

    assert result.mass == pytest.approx(mass)
    assert result.center_of_mass == pytest.approx((1.0 / 3.0, 0.0, -1.0), abs=1.0e-12)
    assert result.inertia_tensor[0][2] == pytest.approx(result.inertia_tensor[2][0], abs=1.0e-12)
    assert abs(result.inertia_tensor[0][2]) > 0.1
    _assert_matrix_close(
        _reconstruct_tensor(result.principal_inertia, result.principal_axes),
        result.inertia_tensor,
        abs=1.0e-9,
    )


def test_regular_prism_square_matches_rotationally_equivalent_box() -> None:
    radius = math.sqrt(0.5)
    height = 2.0
    mass = 5.0
    result = compute_regular_prism_mass_properties(RegularPrismGeometry(4, radius, height), mass)

    expected = diagonal_tensor(compute_box_inertia(mass, (1.0, 1.0, height)))

    assert result.center_of_mass == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-12)
    _assert_matrix_close(result.inertia_tensor, expected, abs=1.0e-12)


def test_frustum_equal_radii_matches_cylinder_inertia_and_center() -> None:
    radius = 0.5
    height = 2.0
    mass = 4.0
    result = compute_frustum_mass_properties(FrustumGeometry(radius, radius, height), mass)

    transverse = mass / 12.0 * (3.0 * radius**2 + height**2)
    axial = 0.5 * mass * radius**2

    assert result.center_of_mass == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-12)
    _assert_matrix_close(result.inertia_tensor, ((transverse, 0.0, 0.0), (0.0, transverse, 0.0), (0.0, 0.0, axial)))


def test_frustum_center_of_mass_moves_toward_larger_radius() -> None:
    geometry = FrustumGeometry(bottom_radius=1.0, top_radius=0.5, height=2.0)

    result = compute_frustum_mass_properties(geometry, mass=6.0)

    assert result.center_of_mass[2] < 0.0
    assert result.inertia_tensor[0][0] == pytest.approx(result.inertia_tensor[1][1])
    assert result.inertia_tensor[0][2] == pytest.approx(0.0, abs=1.0e-12)


def test_full_geometry_mass_properties_dispatches_supported_shapes() -> None:
    assert compute_full_geometry_mass_properties(WedgeGeometry((1.0, 2.0, 3.0)), 1.0).mass == pytest.approx(1.0)
    assert compute_full_geometry_mass_properties(FrustumGeometry(1.0, 0.5, 2.0), 1.0).mass == pytest.approx(1.0)
    assert compute_full_geometry_mass_properties(RegularPrismGeometry(6, 1.0, 2.0), 1.0).mass == pytest.approx(1.0)

    with pytest.raises(InvalidGeometryError, match="full geometry mass properties"):
        compute_full_geometry_mass_properties(BoxGeometry((1.0, 1.0, 1.0)), 1.0)
