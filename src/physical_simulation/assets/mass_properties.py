"""Mass, inertia tensor, and principal-axis properties for rigid bodies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from physical_simulation.validation.asset_validator import _as_float_tuple, _finite_float
from physical_simulation.validation.errors import InvalidMassPropertiesError

Vector3 = tuple[float, float, float]
Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Quaternion = tuple[float, float, float, float]
IDENTITY_AXES: Matrix3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@dataclass(frozen=True)
class MassProperties:
    """Mass, center of mass, full inertia tensor, and principal inertial frame."""

    mass: float
    center_of_mass: Vector3
    inertia_diagonal: Vector3
    inertia_tensor: Optional[Matrix3] = None
    principal_axes: Optional[Matrix3] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mass",
            _finite_float(
                self.mass,
                field_name="mass",
                minimum=0.0,
                strict_minimum=True,
                error_type=InvalidMassPropertiesError,
            ),
        )
        object.__setattr__(
            self,
            "center_of_mass",
            tuple(
                _as_float_tuple(
                    self.center_of_mass,
                    field_name="center_of_mass",
                    length=3,
                    error_type=InvalidMassPropertiesError,
                )
            ),
        )
        inertia_diagonal = tuple(
            _as_float_tuple(
                self.inertia_diagonal,
                field_name="inertia_diagonal",
                length=3,
                strictly_positive=True,
                error_type=InvalidMassPropertiesError,
            )
        )
        inertia_tensor = (
            _diagonal_tensor(inertia_diagonal)
            if self.inertia_tensor is None
            else _validate_matrix3(self.inertia_tensor, field_name="inertia_tensor")
        )
        _validate_symmetric_positive_tensor(inertia_tensor)
        principal_axes = (
            IDENTITY_AXES
            if self.principal_axes is None
            else _validate_matrix3(self.principal_axes, field_name="principal_axes")
        )
        _validate_principal_axes(principal_axes)
        object.__setattr__(self, "inertia_diagonal", inertia_diagonal)
        object.__setattr__(self, "inertia_tensor", inertia_tensor)
        object.__setattr__(self, "principal_axes", principal_axes)

    @classmethod
    def from_full_tensor(
        cls,
        *,
        mass: float,
        center_of_mass: Vector3,
        inertia_tensor: Matrix3,
    ) -> "MassProperties":
        """Create mass properties by diagonalizing a full inertia tensor."""
        from physical_simulation.dynamics.compound_inertia import diagonalize_symmetric_tensor

        principal_inertia, principal_axes = diagonalize_symmetric_tensor(inertia_tensor)
        return cls(
            mass=mass,
            center_of_mass=center_of_mass,
            inertia_diagonal=principal_inertia,
            inertia_tensor=inertia_tensor,
            principal_axes=principal_axes,
        )

    @classmethod
    def from_principal_axes(
        cls,
        *,
        mass: float,
        center_of_mass: Vector3,
        principal_inertia: Vector3,
        principal_axes: Matrix3,
        inertia_tensor: Optional[Matrix3] = None,
    ) -> "MassProperties":
        """Create mass properties from a principal inertial frame."""
        axes = _validate_matrix3(principal_axes, field_name="principal_axes")
        diagonal_values = tuple(
            _as_float_tuple(
                principal_inertia,
                field_name="principal_inertia",
                length=3,
                strictly_positive=True,
                error_type=InvalidMassPropertiesError,
            )
        )
        if inertia_tensor is None:
            inertia_tensor = _multiply_matrix(_multiply_matrix(axes, _diagonal_tensor(diagonal_values)), _transpose(axes))
        return cls(
            mass=mass,
            center_of_mass=center_of_mass,
            inertia_diagonal=diagonal_values,
            inertia_tensor=inertia_tensor,
            principal_axes=axes,
        )

    @property
    def inertial_frame_quaternion(self) -> Quaternion:
        """Return principal inertial frame rotation as a project/MJCF-order quaternion."""
        assert self.principal_axes is not None
        return matrix_to_quaternion(self.principal_axes)

    @property
    def has_non_identity_principal_axes(self) -> bool:
        """Return whether principal axes differ from body-local axes."""
        assert self.principal_axes is not None
        return any(
            abs(self.principal_axes[row][col] - IDENTITY_AXES[row][col]) > 1.0e-9
            for row in range(3)
            for col in range(3)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize mass properties to a JSON-compatible dictionary."""
        assert self.inertia_tensor is not None
        assert self.principal_axes is not None
        return {
            "mass": self.mass,
            "center_of_mass": list(self.center_of_mass),
            "inertia_diagonal": list(self.inertia_diagonal),
            "inertia_tensor": [list(row) for row in self.inertia_tensor],
            "principal_axes": [list(row) for row in self.principal_axes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MassProperties":
        """Deserialize mass properties from a dictionary."""
        if not isinstance(data, dict):
            raise InvalidMassPropertiesError(f"mass_properties data must be a dict; actual value={data!r}")
        return cls(
            mass=data.get("mass"),
            center_of_mass=tuple(data.get("center_of_mass", ())),
            inertia_diagonal=tuple(data.get("inertia_diagonal", ())),
            inertia_tensor=_matrix_from_data(data.get("inertia_tensor")),
            principal_axes=_matrix_from_data(data.get("principal_axes")),
        )


def matrix_to_quaternion(matrix: Matrix3) -> Quaternion:
    """Convert a right-handed orthonormal 3x3 rotation matrix to ``(w, x, y, z)``."""
    m = _validate_matrix3(matrix, field_name="matrix")
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = (
            0.25 * scale,
            (m[2][1] - m[1][2]) / scale,
            (m[0][2] - m[2][0]) / scale,
            (m[1][0] - m[0][1]) / scale,
        )
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        scale = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        quat = (
            (m[2][1] - m[1][2]) / scale,
            0.25 * scale,
            (m[0][1] + m[1][0]) / scale,
            (m[0][2] + m[2][0]) / scale,
        )
    elif m[1][1] > m[2][2]:
        scale = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        quat = (
            (m[0][2] - m[2][0]) / scale,
            (m[0][1] + m[1][0]) / scale,
            0.25 * scale,
            (m[1][2] + m[2][1]) / scale,
        )
    else:
        scale = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        quat = (
            (m[1][0] - m[0][1]) / scale,
            (m[0][2] + m[2][0]) / scale,
            (m[1][2] + m[2][1]) / scale,
            0.25 * scale,
        )
    if quat[0] < 0.0:
        quat = tuple(-value for value in quat)  # type: ignore[assignment]
    from physical_simulation.math import quaternion_normalize

    return quaternion_normalize(quat)


def _matrix_from_data(value: Any) -> Optional[Matrix3]:
    if value is None:
        return None
    return _validate_matrix3(value, field_name="matrix")


def _validate_matrix3(value: Any, *, field_name: str) -> Matrix3:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise InvalidMassPropertiesError(f"{field_name} must be a 3x3 matrix; actual value={value!r}")
    rows = []
    for index, row in enumerate(value):
        rows.append(
            tuple(
                _as_float_tuple(
                    row,
                    field_name=f"{field_name}[{index}]",
                    length=3,
                    error_type=InvalidMassPropertiesError,
                )
            )
        )
    return tuple(rows)  # type: ignore[return-value]


def _validate_symmetric_positive_tensor(tensor: Matrix3) -> None:
    for row in range(3):
        if tensor[row][row] <= 0.0:
            raise InvalidMassPropertiesError(f"inertia_tensor diagonal must be > 0; actual value={tensor!r}")
        for col in range(3):
            if abs(tensor[row][col] - tensor[col][row]) > 1.0e-9:
                raise InvalidMassPropertiesError(f"inertia_tensor must be symmetric; actual value={tensor!r}")


def _validate_principal_axes(axes: Matrix3) -> None:
    columns = [[axes[row][col] for row in range(3)] for col in range(3)]
    for index, column in enumerate(columns):
        norm = math.sqrt(sum(value * value for value in column))
        if abs(norm - 1.0) > 1.0e-7:
            raise InvalidMassPropertiesError(f"principal_axes columns must be unit length; column={index}")
    for first in range(3):
        for second in range(first + 1, 3):
            dot = sum(columns[first][axis] * columns[second][axis] for axis in range(3))
            if abs(dot) > 1.0e-7:
                raise InvalidMassPropertiesError("principal_axes columns must be orthogonal")
    determinant = (
        columns[0][0] * (columns[1][1] * columns[2][2] - columns[1][2] * columns[2][1])
        - columns[1][0] * (columns[0][1] * columns[2][2] - columns[0][2] * columns[2][1])
        + columns[2][0] * (columns[0][1] * columns[1][2] - columns[0][2] * columns[1][1])
    )
    if abs(determinant - 1.0) > 1.0e-7:
        raise InvalidMassPropertiesError(f"principal_axes must be right-handed; determinant={determinant!r}")


def _diagonal_tensor(diagonal: Vector3) -> Matrix3:
    x, y, z = diagonal
    return ((x, 0.0, 0.0), (0.0, y, 0.0), (0.0, 0.0, z))


def _transpose(matrix: Matrix3) -> Matrix3:
    return tuple(tuple(matrix[col][row] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def _multiply_matrix(first: Matrix3, second: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(first[row][k] * second[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]
