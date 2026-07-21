"""Deterministic convex mesh generation for MuJoCo fallback geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

from physical_simulation.assets.geometry import (
    ConeGeometry,
    FrustumGeometry,
    GeometrySpec,
    RegularPrismGeometry,
    WedgeGeometry,
)
from physical_simulation.compilers.errors import UnsupportedPhysicsFeatureError
from physical_simulation.validation.errors import InvalidGeometryError

Vector3 = tuple[float, float, float]
Face = tuple[int, int, int]


@dataclass(frozen=True)
class ConvexMeshSpec:
    """Inline MuJoCo mesh data using local vertices and triangular faces."""

    vertices: tuple[Vector3, ...]
    faces: tuple[Face, ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 4:
            raise InvalidGeometryError("convex mesh fallback requires at least four vertices")
        if not self.faces:
            raise InvalidGeometryError("convex mesh fallback requires at least one face")

    def vertex_attribute(self) -> str:
        """Return MJCF inline vertex attribute text."""
        return " ".join(_format_number(value) for vertex in self.vertices for value in vertex)

    def face_attribute(self) -> str:
        """Return MJCF inline face attribute text."""
        return " ".join(str(index) for face in self.faces for index in face)


def supports_mujoco_mesh_fallback(geometry: GeometrySpec) -> bool:
    """Return whether a geometry can be represented as a deterministic convex mesh."""
    return isinstance(geometry, (WedgeGeometry, ConeGeometry, FrustumGeometry, RegularPrismGeometry))


def geometry_to_convex_mesh(geometry: GeometrySpec) -> ConvexMeshSpec:
    """Generate a deterministic convex mesh fallback for supported geometry specs."""
    if isinstance(geometry, WedgeGeometry):
        return _wedge_mesh(geometry)
    if isinstance(geometry, ConeGeometry):
        return _cone_mesh(geometry, segments=32)
    if isinstance(geometry, FrustumGeometry):
        return _frustum_mesh(geometry, segments=32)
    if isinstance(geometry, RegularPrismGeometry):
        return _regular_prism_mesh(geometry)
    raise UnsupportedPhysicsFeatureError(
        f"unsupported geometry for MuJoCo mesh fallback; geometry={geometry!r}"
    )


def _wedge_mesh(geometry: WedgeGeometry) -> ConvexMeshSpec:
    x, y, z = geometry.size
    hx = x / 2.0
    hy = y / 2.0
    hz = z / 2.0
    vertices = (
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, -hy, hz),
        (-hx, hy, -hz),
        (hx, hy, -hz),
        (hx, hy, hz),
    )
    faces = (
        (0, 1, 2),
        (3, 5, 4),
        (0, 3, 4),
        (0, 4, 1),
        (1, 4, 5),
        (1, 5, 2),
        (2, 5, 3),
        (2, 3, 0),
    )
    return ConvexMeshSpec(vertices=vertices, faces=faces)


def _cone_mesh(geometry: ConeGeometry, *, segments: int) -> ConvexMeshSpec:
    if segments < 3:
        raise InvalidGeometryError(f"cone mesh segments must be >= 3; actual value={segments!r}")
    radius = geometry.radius
    height = geometry.height
    base_z = -height / 2.0
    apex_z = height / 2.0
    base_vertices = _circle_vertices(radius, base_z, segments)
    center_index = len(base_vertices)
    apex_index = center_index + 1
    vertices = base_vertices + ((0.0, 0.0, base_z), (0.0, 0.0, apex_z))
    faces: list[Face] = []
    for index in range(segments):
        next_index = (index + 1) % segments
        faces.append((center_index, next_index, index))
        faces.append((index, next_index, apex_index))
    return ConvexMeshSpec(vertices=vertices, faces=tuple(faces))


def _frustum_mesh(geometry: FrustumGeometry, *, segments: int) -> ConvexMeshSpec:
    if segments < 3:
        raise InvalidGeometryError(f"frustum mesh segments must be >= 3; actual value={segments!r}")
    bottom_z = -geometry.height / 2.0
    top_z = geometry.height / 2.0
    bottom = _circle_vertices(geometry.bottom_radius, bottom_z, segments)
    top = _circle_vertices(geometry.top_radius, top_z, segments)
    bottom_center_index = 2 * segments
    top_center_index = bottom_center_index + 1
    vertices = bottom + top + ((0.0, 0.0, bottom_z), (0.0, 0.0, top_z))
    faces: list[Face] = []
    for index in range(segments):
        next_index = (index + 1) % segments
        top_index = segments + index
        top_next = segments + next_index
        faces.append((bottom_center_index, next_index, index))
        faces.append((top_center_index, top_index, top_next))
        faces.append((index, next_index, top_next))
        faces.append((index, top_next, top_index))
    return ConvexMeshSpec(vertices=vertices, faces=tuple(faces))


def _regular_prism_mesh(geometry: RegularPrismGeometry) -> ConvexMeshSpec:
    segments = geometry.sides
    bottom_z = -geometry.height / 2.0
    top_z = geometry.height / 2.0
    bottom = _circle_vertices(geometry.radius, bottom_z, segments)
    top = _circle_vertices(geometry.radius, top_z, segments)
    bottom_center_index = 2 * segments
    top_center_index = bottom_center_index + 1
    vertices = bottom + top + ((0.0, 0.0, bottom_z), (0.0, 0.0, top_z))
    faces: list[Face] = []
    for index in range(segments):
        next_index = (index + 1) % segments
        top_index = segments + index
        top_next = segments + next_index
        faces.append((bottom_center_index, next_index, index))
        faces.append((top_center_index, top_index, top_next))
        faces.append((index, next_index, top_next))
        faces.append((index, top_next, top_index))
    return ConvexMeshSpec(vertices=vertices, faces=tuple(faces))


def _circle_vertices(radius: float, z: float, segments: int) -> tuple[Vector3, ...]:
    return tuple(
        (
            radius * math.cos(2.0 * math.pi * index / segments),
            radius * math.sin(2.0 * math.pi * index / segments),
            z,
        )
        for index in range(segments)
    )


def _format_number(value: float) -> str:
    if abs(value) < 1.0e-15:
        value = 0.0
    return repr(float(value))
