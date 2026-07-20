"""MuJoCo compiler result types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MuJoCoCompilationResult:
    """Result of compiling a PhysicsSceneSpec into MJCF text."""

    scene_id: str
    mjcf: str
    runtime_body_to_mujoco_name: tuple[tuple[str, str], ...]
    mujoco_geom_to_runtime_body: tuple[tuple[str, str], ...]

    def get_mujoco_body_name(self, runtime_body_id: str) -> str:
        """Return the MuJoCo body name for a runtime body id."""
        for current_runtime_body_id, mujoco_name in self.runtime_body_to_mujoco_name:
            if current_runtime_body_id == runtime_body_id:
                return mujoco_name
        raise KeyError(f"runtime_body_id not found in MuJoCoCompilationResult: {runtime_body_id!r}")

    def get_runtime_body_for_geom(self, geom_name: str) -> str:
        """Return the runtime body id for a MuJoCo collision geom name."""
        for current_geom_name, runtime_body_id in self.mujoco_geom_to_runtime_body:
            if current_geom_name == geom_name:
                return runtime_body_id
        raise KeyError(f"geom_name not found in MuJoCoCompilationResult: {geom_name!r}")
