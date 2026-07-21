import math

import pytest

from physical_simulation.assets import BoxGeometry, SphereGeometry, Transform
from physical_simulation.dynamics import (
    CompoundInertiaComponent,
    compute_box_inertia,
    compute_compound_mass_properties,
    diagonalize_symmetric_tensor,
)
from physical_simulation.dynamics.compound_inertia import (
    Matrix3,
    diagonal_tensor,
    multiply_matrix,
    rotation_matrix_from_quaternion,
    transpose,
)
from physical_simulation.validation.errors import InvalidMassPropertiesError


def _qz(degrees: float) -> tuple[float, float, float, float]:
    radians = math.radians(degrees)
    return (math.cos(radians / 2.0), 0.0, 0.0, math.sin(radians / 2.0))


def _assert_matrix_close(actual: Matrix3, expected: Matrix3, *, abs: float = 1.0e-12) -> None:
    for row in range(3):
        assert actual[row] == pytest.approx(expected[row], abs=abs)


def _reconstruct_tensor(principal: tuple[float, float, float], axes: Matrix3) -> Matrix3:
    return multiply_matrix(multiply_matrix(axes, diagonal_tensor(principal)), transpose(axes))


def test_compound_center_of_mass_uses_component_masses() -> None:
    components = (
        CompoundInertiaComponent(
            SphereGeometry(0.1),
            1.0,
            Transform(position=(-1.0, 0.0, 0.0)),
        ),
        CompoundInertiaComponent(
            SphereGeometry(0.1),
            3.0,
            Transform(position=(1.0, 0.0, 0.0)),
        ),
    )

    result = compute_compound_mass_properties(components)

    assert result.mass == pytest.approx(4.0)
    assert result.center_of_mass == pytest.approx((0.5, 0.0, 0.0))


def test_parallel_axis_for_two_spheres_matches_expected_tensor() -> None:
    components = (
        CompoundInertiaComponent(SphereGeometry(0.1), 2.0, Transform(position=(-1.0, 0.0, 0.0))),
        CompoundInertiaComponent(SphereGeometry(0.1), 2.0, Transform(position=(1.0, 0.0, 0.0))),
    )

    result = compute_compound_mass_properties(components)

    sphere_inertia = 2.0 / 5.0 * 2.0 * 0.1**2
    expected = (
        (2.0 * sphere_inertia, 0.0, 0.0),
        (0.0, 2.0 * (sphere_inertia + 2.0 * 1.0**2), 0.0),
        (0.0, 0.0, 2.0 * (sphere_inertia + 2.0 * 1.0**2)),
    )
    _assert_matrix_close(result.inertia_tensor, expected)


def test_rotated_box_produces_full_tensor_with_products_of_inertia() -> None:
    geometry = BoxGeometry((1.0, 2.0, 3.0))
    mass = 2.0
    rotation = _qz(45.0)
    result = compute_compound_mass_properties(
        (CompoundInertiaComponent(geometry, mass, Transform(rotation=rotation)),)
    )

    diagonal = diagonal_tensor(compute_box_inertia(mass, geometry.size))
    matrix = rotation_matrix_from_quaternion(rotation)
    expected = multiply_matrix(multiply_matrix(matrix, diagonal), transpose(matrix))

    _assert_matrix_close(result.inertia_tensor, expected)
    assert abs(result.inertia_tensor[0][1]) > 0.1
    assert result.inertia_tensor[0][1] == pytest.approx(result.inertia_tensor[1][0])


def test_principal_axis_decomposition_reconstructs_full_tensor() -> None:
    tensor = (
        (5.0, -1.0, 0.75),
        (-1.0, 3.0, 0.25),
        (0.75, 0.25, 2.0),
    )

    principal, axes = diagonalize_symmetric_tensor(tensor)
    reconstructed = _reconstruct_tensor(principal, axes)

    _assert_matrix_close(reconstructed, tensor, abs=1.0e-9)


def test_compound_mass_properties_can_be_viewed_as_existing_mass_properties() -> None:
    result = compute_compound_mass_properties(
        (
            CompoundInertiaComponent(BoxGeometry((1.0, 2.0, 3.0)), 2.0),
            CompoundInertiaComponent(BoxGeometry((0.5, 0.5, 0.5)), 1.0, Transform(position=(1.0, 0.0, 0.0))),
        )
    )

    mass_properties = result.to_mass_properties()

    assert mass_properties.mass == pytest.approx(result.mass)
    assert mass_properties.center_of_mass == pytest.approx(result.center_of_mass)
    assert mass_properties.inertia_diagonal == pytest.approx(result.principal_inertia)


def test_component_rejects_unbaked_scale() -> None:
    with pytest.raises(InvalidMassPropertiesError, match="bake scale"):
        CompoundInertiaComponent(
            BoxGeometry((1.0, 1.0, 1.0)),
            1.0,
            Transform(scale=(2.0, 1.0, 1.0)),
        )


def test_empty_compound_rejected() -> None:
    with pytest.raises(InvalidMassPropertiesError, match="at least one"):
        compute_compound_mass_properties(())
