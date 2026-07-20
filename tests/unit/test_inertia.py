import pytest

from physical_simulation.assets import BoxGeometry
from physical_simulation.dynamics import (
    compute_box_inertia,
    compute_capsule_inertia,
    compute_cylinder_inertia,
    compute_mass_from_density,
    compute_sphere_inertia,
)
from physical_simulation.validation.errors import InvalidMassPropertiesError


def test_box_inertia() -> None:
    assert compute_box_inertia(12.0, (2.0, 3.0, 4.0)) == pytest.approx((25.0, 20.0, 13.0))


def test_sphere_inertia() -> None:
    assert compute_sphere_inertia(10.0, 2.0) == pytest.approx((16.0, 16.0, 16.0))


def test_cylinder_inertia() -> None:
    assert compute_cylinder_inertia(12.0, 2.0, 3.0) == pytest.approx((21.0, 21.0, 24.0))


def test_capsule_inertia_is_positive() -> None:
    inertia = compute_capsule_inertia(3.0, 0.2, 1.0)
    assert all(component > 0.0 for component in inertia)


def test_longer_capsule_increases_transverse_inertia() -> None:
    short = compute_capsule_inertia(3.0, 0.2, 0.5)
    long = compute_capsule_inertia(3.0, 0.2, 2.0)
    assert long[0] > short[0]


def test_invalid_mass_raises() -> None:
    with pytest.raises(InvalidMassPropertiesError, match="mass"):
        compute_box_inertia(0.0, (1.0, 1.0, 1.0))


def test_mass_from_density() -> None:
    assert compute_mass_from_density(BoxGeometry((2.0, 3.0, 4.0)), 10.0) == pytest.approx(240.0)
