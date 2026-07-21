"""Compatibility exports for MuJoCo convex mesh fallback generation."""

from physical_simulation.collision.convex_mesh import (
    ConvexMeshSpec,
    geometry_to_convex_mesh,
    supports_mujoco_mesh_fallback,
)

__all__ = ["ConvexMeshSpec", "geometry_to_convex_mesh", "supports_mujoco_mesh_fallback"]
