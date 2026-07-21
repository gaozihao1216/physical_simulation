import math

import pytest

from physical_simulation.assets import MassProperties
from physical_simulation.assets.mass_properties import matrix_to_quaternion
from physical_simulation.validation.errors import InvalidMassPropertiesError


def _qz(degrees: float) -> tuple[float, float, float, float]:
    radians = math.radians(degrees)
    return (math.cos(radians / 2.0), 0.0, 0.0, math.sin(radians / 2.0))


def _rz(degrees: float):
    radians = math.radians(degrees)
    c = math.cos(radians)
    s = math.sin(radians)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def test_legacy_diagonal_mass_properties_populates_identity_frame() -> None:
    props = MassProperties(1.0, (0.0, 0.0, 0.0), (1.0, 2.0, 3.0))

    assert props.inertia_tensor == ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0))
    assert props.principal_axes == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert not props.has_non_identity_principal_axes
    assert props.inertial_frame_quaternion == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_full_tensor_constructor_diagonalizes_and_preserves_tensor() -> None:
    tensor = ((2.5, -1.5, 0.0), (-1.5, 2.5, 0.0), (0.0, 0.0, 4.0))

    props = MassProperties.from_full_tensor(
        mass=2.0,
        center_of_mass=(0.1, 0.2, 0.3),
        inertia_tensor=tensor,
    )

    assert props.mass == pytest.approx(2.0)
    assert props.center_of_mass == pytest.approx((0.1, 0.2, 0.3))
    assert props.inertia_tensor == tensor
    assert props.inertia_diagonal == pytest.approx((1.0, 4.0, 4.0))
    assert props.has_non_identity_principal_axes


def test_principal_axes_constructor_reconstructs_full_tensor() -> None:
    axes = _rz(45.0)
    props = MassProperties.from_principal_axes(
        mass=1.0,
        center_of_mass=(0.0, 0.0, 0.0),
        principal_inertia=(1.0, 4.0, 5.0),
        principal_axes=axes,
    )

    assert props.inertia_tensor[0][1] == pytest.approx(props.inertia_tensor[1][0])
    assert abs(props.inertia_tensor[0][1]) > 1.0
    assert props.inertial_frame_quaternion == pytest.approx(_qz(45.0))


def test_mass_properties_serialization_round_trip_keeps_extended_fields() -> None:
    props = MassProperties.from_principal_axes(
        mass=1.0,
        center_of_mass=(0.0, 0.0, 0.0),
        principal_inertia=(1.0, 2.0, 3.0),
        principal_axes=_rz(30.0),
    )

    assert MassProperties.from_dict(props.to_dict()) == props


def test_invalid_full_tensor_and_axes_raise() -> None:
    with pytest.raises(InvalidMassPropertiesError, match="symmetric"):
        MassProperties(
            1.0,
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            inertia_tensor=((1.0, 0.5, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)),
        )
    with pytest.raises(InvalidMassPropertiesError, match="unit length"):
        MassProperties(
            1.0,
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            principal_axes=((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )


def test_matrix_to_quaternion_handles_identity_and_z_rotation() -> None:
    assert matrix_to_quaternion(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))) == pytest.approx(
        (1.0, 0.0, 0.0, 0.0)
    )
    assert matrix_to_quaternion(_rz(90.0)) == pytest.approx(_qz(90.0))
