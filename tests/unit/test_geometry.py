import math

import pytest

from physical_simulation.assets import (
    BoxGeometry,
    CapsuleGeometry,
    ConeGeometry,
    CylinderGeometry,
    EllipsoidGeometry,
    FrustumGeometry,
    RegularPrismGeometry,
    SphericalCapGeometry,
    SphereGeometry,
    WedgeGeometry,
    geometry_from_dict,
)
from physical_simulation.validation.errors import InvalidGeometryError


def test_geometry_volumes() -> None:
    assert BoxGeometry((2.0, 3.0, 4.0)).volume() == pytest.approx(24.0)
    assert SphereGeometry(2.0).volume() == pytest.approx(4.0 / 3.0 * math.pi * 8.0)
    assert CylinderGeometry(2.0, 3.0).volume() == pytest.approx(math.pi * 4.0 * 3.0)
    assert CapsuleGeometry(1.0, 2.0).volume() == pytest.approx(math.pi * 2.0 + 4.0 / 3.0 * math.pi)
    assert WedgeGeometry((2.0, 3.0, 4.0)).volume() == pytest.approx(12.0)
    assert ConeGeometry(2.0, 3.0).volume() == pytest.approx(math.pi * 4.0)
    assert FrustumGeometry(2.0, 1.0, 3.0).volume() == pytest.approx(7.0 * math.pi)
    assert EllipsoidGeometry((1.0, 2.0, 3.0)).volume() == pytest.approx(8.0 * math.pi)
    assert SphericalCapGeometry(2.0, 0.5).volume() == pytest.approx(math.pi * 0.25 * 5.5 / 3.0)
    assert RegularPrismGeometry(6, 1.0, 2.0).volume() == pytest.approx(3.0 * math.sqrt(3.0))


def test_invalid_dimensions_raise() -> None:
    with pytest.raises(InvalidGeometryError, match="size"):
        BoxGeometry((1.0, -1.0, 1.0))
    with pytest.raises(InvalidGeometryError, match="radius"):
        SphereGeometry(0.0)
    with pytest.raises(InvalidGeometryError, match="height"):
        SphericalCapGeometry(1.0, 3.0)
    with pytest.raises(InvalidGeometryError, match="sides"):
        RegularPrismGeometry(2, 1.0, 1.0)


def test_unknown_shape_type_raises() -> None:
    with pytest.raises(InvalidGeometryError, match="shape_type"):
        geometry_from_dict({"shape_type": "mesh"})


def test_dict_round_trip() -> None:
    geometries = [
        BoxGeometry((1.0, 2.0, 3.0)),
        SphereGeometry(0.5),
        CylinderGeometry(0.5, 2.0),
        CapsuleGeometry(0.5, 2.0),
        WedgeGeometry((1.0, 2.0, 3.0)),
        ConeGeometry(0.5, 2.0),
        FrustumGeometry(0.5, 0.25, 2.0),
        EllipsoidGeometry((0.5, 1.0, 1.5)),
        SphericalCapGeometry(1.0, 0.25),
        RegularPrismGeometry(5, 0.5, 2.0),
    ]
    for geometry in geometries:
        assert geometry_from_dict(geometry.to_dict()) == geometry
