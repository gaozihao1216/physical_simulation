"""Backend compiler entry points."""

from physical_simulation.compilers.errors import (
    MuJoCoCompilationError,
    PhysicsCompilationError,
    UnsupportedAssetStructureError,
    UnsupportedPhysicsFeatureError,
)
from physical_simulation.compilers.mujoco_compiler import (
    MUJOCO_ALL_COLLISION_BITS,
    MUJOCO_EXPLICIT_PAIR_CONDIM,
    MUJOCO_EXPLICIT_PAIR_GAP,
    MUJOCO_EXPLICIT_PAIR_MARGIN,
    MUJOCO_ROLLING_FRICTION,
    MUJOCO_TORSIONAL_FRICTION,
    MuJoCoCompiler,
    collision_pair_enabled,
    geometry_to_mujoco,
    make_mujoco_name,
)
from physical_simulation.compilers.mujoco_types import CompiledColliderMetadata, MuJoCoCompilationResult

__all__ = [
    "CompiledColliderMetadata",
    "MuJoCoCompiler",
    "MuJoCoCompilationResult",
    "PhysicsCompilationError",
    "UnsupportedPhysicsFeatureError",
    "UnsupportedAssetStructureError",
    "MuJoCoCompilationError",
    "MUJOCO_ALL_COLLISION_BITS",
    "MUJOCO_EXPLICIT_PAIR_CONDIM",
    "MUJOCO_EXPLICIT_PAIR_MARGIN",
    "MUJOCO_EXPLICIT_PAIR_GAP",
    "MUJOCO_TORSIONAL_FRICTION",
    "MUJOCO_ROLLING_FRICTION",
    "geometry_to_mujoco",
    "collision_pair_enabled",
    "make_mujoco_name",
]
