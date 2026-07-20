import math

import pytest

from physical_simulation.assets import (
    BoxGeometry,
    CapsuleGeometry,
    CylinderGeometry,
    SphereGeometry,
    geometry_from_dict,
)
from physical_simulation.validation.errors import InvalidGeometryError


def test_geometry_volumes() -> None:
    assert BoxGeometry((2.0, 3.0, 4.0)).volume() == pytest.approx(24.0)
    assert SphereGeometry(2.0).volume() == pytest.approx(4.0 / 3.0 * math.pi * 8.0)
    assert CylinderGeometry(2.0, 3.0).volume() == pytest.approx(math.pi * 4.0 * 3.0)
    assert CapsuleGeometry(1.0, 2.0).volume() == pytest.approx(math.pi * 2.0 + 4.0 / 3.0 * math.pi)


def test_invalid_dimensions_raise() -> None:
    with pytest.raises(InvalidGeometryError, match="size"):
        BoxGeometry((1.0, -1.0, 1.0))
    with pytest.raises(InvalidGeometryError, match="radius"):
        SphereGeometry(0.0)


def test_unknown_shape_type_raises() -> None:
    with pytest.raises(InvalidGeometryError, match="shape_type"):
        geometry_from_dict({"shape_type": "mesh"})


def test_dict_round_trip() -> None:
    geometries = [
        BoxGeometry((1.0, 2.0, 3.0)),
        SphereGeometry(0.5),
        CylinderGeometry(0.5, 2.0),
        CapsuleGeometry(0.5, 2.0),
    ]
    for geometry in geometries:
        assert geometry_from_dict(geometry.to_dict()) == geometry
