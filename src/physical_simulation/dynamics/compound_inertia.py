"""Compound rigid-body inertia calculations using full 3x3 tensors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from physical_simulation.assets.geometry import GeometrySpec
from physical_simulation.assets.mass_properties import MassProperties
from physical_simulation.assets.transform import Transform
from physical_simulation.dynamics.inertia import compute_inertia, compute_mass_from_density
from physical_simulation.validation.asset_validator import _finite_float
from physical_simulation.validation.errors import InvalidMassPropertiesError

Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class CompoundInertiaComponent:
    """A primitive mass element placed in a compound rigid body's local frame."""

    geometry: GeometrySpec
    mass: float
    transform: Transform = Transform.identity()

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
        from physical_simulation.assets.scale_baking import is_unit_scale

        if not is_unit_scale(self.transform.scale):
            raise InvalidMassPropertiesError(
                "component transform.scale must be unit scale for compound inertia; "
                f"actual value={self.transform.scale!r}; bake scale into geometry before inertia calculation"
            )

    @classmethod
    def from_density(
        cls,
        geometry: GeometrySpec,
        density: float,
        transform: Transform = Transform.identity(),
    ) -> "CompoundInertiaComponent":
        """Create a component whose mass is computed from geometry volume and density."""
        return cls(geometry=geometry, mass=compute_mass_from_density(geometry, density), transform=transform)


@dataclass(frozen=True)
class CompoundMassProperties:
    """Full mass properties for a compound rigid body in body-local coordinates."""

    mass: float
    center_of_mass: Vector3
    inertia_tensor: Matrix3
    principal_inertia: Vector3
    principal_axes: Matrix3

    def to_mass_properties(self) -> MassProperties:
        """Return the existing diagonal MassProperties view in the principal frame."""
        return MassProperties(
            mass=self.mass,
            center_of_mass=self.center_of_mass,
            inertia_diagonal=self.principal_inertia,
        )


def zero_matrix() -> Matrix3:
    """Return a 3x3 zero matrix."""
    return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def diagonal_tensor(diagonal: Vector3) -> Matrix3:
    """Return a diagonal inertia tensor from three diagonal components."""
    x, y, z = diagonal
    return ((x, 0.0, 0.0), (0.0, y, 0.0), (0.0, 0.0, z))


def rotation_matrix_from_quaternion(quaternion: tuple[float, float, float, float]) -> Matrix3:
    """Convert a project-order quaternion ``(w, x, y, z)`` to a rotation matrix."""
    from physical_simulation.math import quaternion_normalize

    w, x, y, z = quaternion_normalize(quaternion)
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def add_matrix(first: Matrix3, second: Matrix3) -> Matrix3:
    """Add two 3x3 matrices."""
    return tuple(
        tuple(first[row][col] + second[row][col] for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def transpose(matrix: Matrix3) -> Matrix3:
    """Transpose a 3x3 matrix."""
    return tuple(tuple(matrix[col][row] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def multiply_matrix(first: Matrix3, second: Matrix3) -> Matrix3:
    """Multiply two 3x3 matrices."""
    return tuple(
        tuple(sum(first[row][k] * second[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def rotate_inertia_tensor(tensor: Matrix3, rotation: tuple[float, float, float, float]) -> Matrix3:
    """Rotate an inertia tensor into the parent frame with ``R I R^T``."""
    matrix = rotation_matrix_from_quaternion(rotation)
    return multiply_matrix(multiply_matrix(matrix, tensor), transpose(matrix))


def translate_inertia_tensor(tensor: Matrix3, mass: float, offset: Vector3) -> Matrix3:
    """Apply the parallel-axis theorem for an offset from the target center of mass."""
    m = _finite_float(
        mass,
        field_name="mass",
        minimum=0.0,
        strict_minimum=True,
        error_type=InvalidMassPropertiesError,
    )
    dx, dy, dz = offset
    distance_squared = dx * dx + dy * dy + dz * dz
    parallel_axis = (
        (m * (distance_squared - dx * dx), -m * dx * dy, -m * dx * dz),
        (-m * dy * dx, m * (distance_squared - dy * dy), -m * dy * dz),
        (-m * dz * dx, -m * dz * dy, m * (distance_squared - dz * dz)),
    )
    return add_matrix(tensor, parallel_axis)


def diagonalize_symmetric_tensor(tensor: Matrix3) -> tuple[Vector3, Matrix3]:
    """Return sorted principal inertia values and principal axes for a symmetric tensor.

    The returned matrix stores principal axes as columns expressed in the input
    tensor's frame. Axis signs are canonicalized for deterministic output.
    """
    a = [[float(tensor[row][col]) for col in range(3)] for row in range(3)]
    _validate_symmetric_matrix(a)
    vectors = [[1.0 if row == col else 0.0 for col in range(3)] for row in range(3)]

    for _ in range(64):
        p, q = _largest_off_diagonal_index(a)
        if abs(a[p][q]) <= 1.0e-12:
            break
        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]
        angle = 0.5 * math.atan2(2.0 * apq, aqq - app)
        c = math.cos(angle)
        s = math.sin(angle)

        for k in range(3):
            if k == p or k == q:
                continue
            akp = a[k][p]
            akq = a[k][q]
            a[k][p] = a[p][k] = c * akp - s * akq
            a[k][q] = a[q][k] = s * akp + c * akq

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = a[q][p] = 0.0

        for k in range(3):
            vkp = vectors[k][p]
            vkq = vectors[k][q]
            vectors[k][p] = c * vkp - s * vkq
            vectors[k][q] = s * vkp + c * vkq

    ordered = sorted((a[index][index], index) for index in range(3))
    principal = tuple(max(value, 0.0) for value, _ in ordered)
    axes_columns = [[vectors[row][index] for row in range(3)] for _, index in ordered]
    axes_columns = [_canonicalize_axis(axis) for axis in axes_columns]
    if _determinant_from_columns(axes_columns) < 0.0:
        axes_columns[-1] = [-value for value in axes_columns[-1]]
    axes = tuple(
        tuple(axes_columns[col][row] for col in range(3))
        for row in range(3)
    )
    return principal, axes  # type: ignore[return-value]


def compute_compound_mass_properties(
    components: Iterable[CompoundInertiaComponent],
) -> CompoundMassProperties:
    """Compute full tensor mass properties for primitive components.

    Each component transform is interpreted in the compound rigid body's local
    frame. Scale must already be baked into the component geometry.
    """
    component_tuple = tuple(components)
    if not component_tuple:
        raise InvalidMassPropertiesError("components must contain at least one mass component")

    total_mass = sum(component.mass for component in component_tuple)
    if total_mass <= 0.0:
        raise InvalidMassPropertiesError("total compound mass must be > 0")
    center_of_mass = tuple(
        sum(component.mass * component.transform.position[axis] for component in component_tuple) / total_mass
        for axis in range(3)
    )

    total_tensor = zero_matrix()
    for component in component_tuple:
        local_diagonal = compute_inertia(component.geometry, component.mass)
        local_tensor = diagonal_tensor(local_diagonal)
        rotated_tensor = rotate_inertia_tensor(local_tensor, component.transform.rotation)
        offset = tuple(component.transform.position[axis] - center_of_mass[axis] for axis in range(3))
        shifted_tensor = translate_inertia_tensor(rotated_tensor, component.mass, offset)  # type: ignore[arg-type]
        total_tensor = add_matrix(total_tensor, shifted_tensor)

    principal_inertia, principal_axes = diagonalize_symmetric_tensor(total_tensor)
    return CompoundMassProperties(
        mass=total_mass,
        center_of_mass=center_of_mass,  # type: ignore[arg-type]
        inertia_tensor=_symmetrize(total_tensor),
        principal_inertia=principal_inertia,
        principal_axes=principal_axes,
    )


def _largest_off_diagonal_index(matrix: list[list[float]]) -> tuple[int, int]:
    candidates = ((0, 1), (0, 2), (1, 2))
    return max(candidates, key=lambda pair: abs(matrix[pair[0]][pair[1]]))


def _validate_symmetric_matrix(matrix: list[list[float]]) -> None:
    for row in range(3):
        for col in range(3):
            if not math.isfinite(matrix[row][col]):
                raise InvalidMassPropertiesError(f"inertia tensor must be finite; actual value={matrix!r}")
            if abs(matrix[row][col] - matrix[col][row]) > 1.0e-9:
                raise InvalidMassPropertiesError(f"inertia tensor must be symmetric; actual value={matrix!r}")


def _canonicalize_axis(axis: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 1.0e-12:
        raise InvalidMassPropertiesError(f"principal axis norm must be > 1e-12; actual value={axis!r}")
    normalized = [value / norm for value in axis]
    dominant = max(range(3), key=lambda index: abs(normalized[index]))
    if normalized[dominant] < 0.0:
        return [-value for value in normalized]
    return normalized


def _determinant_from_columns(columns: list[list[float]]) -> float:
    a, b, c = columns
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - b[0] * (a[1] * c[2] - a[2] * c[1])
        + c[0] * (a[1] * b[2] - a[2] * b[1])
    )


def _symmetrize(matrix: Matrix3) -> Matrix3:
    return tuple(
        tuple(0.5 * (matrix[row][col] + matrix[col][row]) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]
