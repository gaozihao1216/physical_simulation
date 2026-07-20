import pytest

from physical_simulation.assets import BoxGeometry, CapsuleGeometry, CylinderGeometry, SphereGeometry
from physical_simulation.compilers import UnsupportedPhysicsFeatureError, geometry_to_mujoco


def test_box_maps_full_size_to_half_extents() -> None:
    assert geometry_to_mujoco(BoxGeometry((2.0, 4.0, 6.0))) == ("box", (1.0, 2.0, 3.0))


def test_sphere_maps_radius() -> None:
    assert geometry_to_mujoco(SphereGeometry(0.25)) == ("sphere", (0.25,))


def test_cylinder_maps_full_height_to_half_height() -> None:
    assert geometry_to_mujoco(CylinderGeometry(0.25, 2.0)) == ("cylinder", (0.25, 1.0))


def test_capsule_maps_cylinder_length_to_half_length() -> None:
    assert geometry_to_mujoco(CapsuleGeometry(0.25, 2.0)) == ("capsule", (0.25, 1.0))


def test_unknown_geometry_raises() -> None:
    with pytest.raises(UnsupportedPhysicsFeatureError):
        geometry_to_mujoco(object())  # type: ignore[arg-type]
