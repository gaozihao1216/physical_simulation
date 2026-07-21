"""Collision authoring utilities and deterministic convex mesh helpers.

TODO: Add primitive fitting and convex decomposition.
"""

from physical_simulation.collision.convex_mesh import (
    ConvexMeshSpec,
    cone_to_convex_mesh,
    frustum_to_convex_mesh,
    geometry_to_convex_mesh,
    regular_prism_to_convex_mesh,
    supports_mujoco_mesh_fallback,
    wedge_to_convex_mesh,
)

__all__ = [
    "ConvexMeshSpec",
    "geometry_to_convex_mesh",
    "supports_mujoco_mesh_fallback",
    "wedge_to_convex_mesh",
    "cone_to_convex_mesh",
    "frustum_to_convex_mesh",
    "regular_prism_to_convex_mesh",
]
