"""MuJoCo model loading and ID mapping backend.

Phase 2C1 loads MJCF into MuJoCo and builds runtime ID mappings only. It does
not implement reset, step, state extraction, contacts, or forces yet.
"""

from __future__ import annotations

from typing import Any, Optional

from physical_simulation.backends.base import PhysicsBackend
from physical_simulation.backends.errors import (
    BackendNotLoadedError,
    MuJoCoModelLoadingError,
    MuJoCoUnavailableError,
    UnknownRuntimeBodyError,
    UnknownRuntimeGeomError,
)
from physical_simulation.compilers import MuJoCoCompilationResult, MuJoCoCompiler
from physical_simulation.scene import PhysicsSceneSpec
from physical_simulation.validation.asset_validator import validate_physics_scene


def _import_mujoco():
    """Import the optional MuJoCo package lazily."""
    try:
        import mujoco
    except ImportError as exc:
        raise MuJoCoUnavailableError(
            "MuJoCo backend requires the optional 'mujoco' dependency. "
            'Install it with: pip install -e ".[mujoco]"'
        ) from exc
    return mujoco


def _resolve_body_id(mujoco_module: Any, model: Any, body_name: str) -> int:
    """Resolve a MuJoCo body name to a numeric body id."""
    body_id = int(mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_BODY, body_name))
    if body_id < 0:
        raise UnknownRuntimeBodyError(
            f"MuJoCo body name was not found in loaded model; body_name={body_name!r}"
        )
    return body_id


def _resolve_geom_id(mujoco_module: Any, model: Any, geom_name: str) -> int:
    """Resolve a MuJoCo geom name to a numeric geom id."""
    geom_id = int(mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_GEOM, geom_name))
    if geom_id < 0:
        raise UnknownRuntimeGeomError(
            f"MuJoCo geom name was not found in loaded model; geom_name={geom_name!r}"
        )
    return geom_id


class MuJoCoBackend(PhysicsBackend):
    """MuJoCo loader for Phase 2C1 model and ID mapping validation."""

    def __init__(self, compiler: Optional[MuJoCoCompiler] = None) -> None:
        self._compiler = compiler or MuJoCoCompiler()
        self._scene: Optional[PhysicsSceneSpec] = None
        self._compilation_result: Optional[MuJoCoCompilationResult] = None
        self._model: Any = None
        self._data: Any = None
        self._runtime_body_to_mj_body_id: dict[str, int] = {}
        self._mj_body_id_to_runtime_body: dict[int, str] = {}
        self._mj_geom_id_to_runtime_body: dict[int, str] = {}
        self._runtime_body_to_collision_geom_ids: dict[str, tuple[int, ...]] = {}
        self._loaded = False
        self._closed = False

    @property
    def is_loaded(self) -> bool:
        """Return whether a scene is currently loaded."""
        return self._loaded

    @property
    def scene(self) -> Optional[PhysicsSceneSpec]:
        """Return the currently loaded scene, if any."""
        return self._scene

    @property
    def compilation_result(self) -> Optional[MuJoCoCompilationResult]:
        """Return the current compilation result, if any."""
        return self._compilation_result

    @property
    def mjcf(self) -> str:
        """Return loaded MJCF text."""
        if not self._loaded or self._compilation_result is None:
            raise BackendNotLoadedError("mjcf is only available after load_scene() succeeds")
        return self._compilation_result.mjcf

    def load_scene(self, scene: PhysicsSceneSpec) -> None:
        """Compile and load a scene into MuJoCo, then build private ID mappings."""
        validate_physics_scene(scene)
        compilation_result = self._compiler.compile(scene)
        mujoco = _import_mujoco()
        try:
            model = mujoco.MjModel.from_xml_string(compilation_result.mjcf)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            runtime_body_to_mj_body_id, mj_body_id_to_runtime_body = self._build_body_mappings(
                mujoco,
                model,
                compilation_result,
            )
            mj_geom_id_to_runtime_body, runtime_body_to_collision_geom_ids = self._build_geom_mappings(
                mujoco,
                model,
                compilation_result,
            )
        except (UnknownRuntimeBodyError, UnknownRuntimeGeomError):
            raise
        except Exception as exc:
            raise MuJoCoModelLoadingError(
                f"failed to load MJCF into MuJoCo model; scene_id={scene.scene_id!r}; original_error={exc}"
            ) from exc

        self._scene = scene
        self._compilation_result = compilation_result
        self._model = model
        self._data = data
        self._runtime_body_to_mj_body_id = runtime_body_to_mj_body_id
        self._mj_body_id_to_runtime_body = mj_body_id_to_runtime_body
        self._mj_geom_id_to_runtime_body = mj_geom_id_to_runtime_body
        self._runtime_body_to_collision_geom_ids = runtime_body_to_collision_geom_ids
        self._loaded = True
        self._closed = False

    def _build_body_mappings(
        self,
        mujoco_module: Any,
        model: Any,
        compilation_result: MuJoCoCompilationResult,
    ) -> tuple[dict[str, int], dict[int, str]]:
        runtime_to_id: dict[str, int] = {}
        id_to_runtime: dict[int, str] = {}
        for runtime_body_id, body_name in compilation_result.runtime_body_to_mujoco_name:
            body_id = _resolve_body_id(mujoco_module, model, body_name)
            if runtime_body_id in runtime_to_id:
                raise MuJoCoModelLoadingError(f"duplicate runtime body mapping; runtime_body_id={runtime_body_id!r}")
            if body_id in id_to_runtime:
                raise MuJoCoModelLoadingError(
                    f"duplicate MuJoCo body numeric ID mapping; body_id={body_id!r}, body_name={body_name!r}"
                )
            runtime_to_id[runtime_body_id] = body_id
            id_to_runtime[body_id] = runtime_body_id
        return runtime_to_id, id_to_runtime

    def _build_geom_mappings(
        self,
        mujoco_module: Any,
        model: Any,
        compilation_result: MuJoCoCompilationResult,
    ) -> tuple[dict[int, str], dict[str, tuple[int, ...]]]:
        geom_to_runtime: dict[int, str] = {}
        runtime_to_geoms: dict[str, list[int]] = {}
        for geom_name, runtime_body_id in compilation_result.mujoco_geom_to_runtime_body:
            geom_id = _resolve_geom_id(mujoco_module, model, geom_name)
            if geom_id in geom_to_runtime:
                raise MuJoCoModelLoadingError(
                    f"duplicate MuJoCo geom numeric ID mapping; geom_id={geom_id!r}, geom_name={geom_name!r}"
                )
            geom_to_runtime[geom_id] = runtime_body_id
            runtime_to_geoms.setdefault(runtime_body_id, []).append(geom_id)
        return geom_to_runtime, {
            runtime_body_id: tuple(geom_ids)
            for runtime_body_id, geom_ids in runtime_to_geoms.items()
        }

    def _require_runtime_body_id(self, runtime_body_id: str) -> int:
        """Return private MuJoCo body id for a runtime body id."""
        if runtime_body_id not in self._runtime_body_to_mj_body_id:
            raise UnknownRuntimeBodyError(
                f"runtime body ID is not loaded; runtime_body_id={runtime_body_id!r}"
            )
        return self._runtime_body_to_mj_body_id[runtime_body_id]

    def _require_runtime_geom_id(self, geom_id: int) -> str:
        """Return runtime body id for a private MuJoCo collision geom id."""
        if geom_id not in self._mj_geom_id_to_runtime_body:
            raise UnknownRuntimeGeomError(f"MuJoCo geom ID is not mapped; geom_id={geom_id!r}")
        return self._mj_geom_id_to_runtime_body[geom_id]

    def reset(self, seed: Optional[int] = None) -> None:
        """Phase 2C2 will implement public reset behavior."""
        raise NotImplementedError("MuJoCoBackend.reset() will be implemented in Phase 2C2.")

    def step(self, dt: float) -> None:
        """Phase 2C2 will implement stepping."""
        raise NotImplementedError("MuJoCoBackend.step() will be implemented in Phase 2C2.")

    def get_body_state(self, body_id: str) -> Any:
        """Phase 2C2 will implement body state extraction."""
        raise NotImplementedError("MuJoCoBackend.get_body_state() will be implemented in Phase 2C2.")

    def get_joint_state(self, joint_id: str) -> Any:
        """Joint state extraction is not implemented in Phase 2C1."""
        raise NotImplementedError("MuJoCoBackend.get_joint_state() is not implemented in Phase 2C1.")

    def get_contacts(self) -> Any:
        """Contact extraction is not implemented until a later phase."""
        raise NotImplementedError("MuJoCoBackend.get_contacts() is not implemented in Phase 2C1.")

    def apply_force(self, body_id: str, force: Any, point: Optional[Any] = None) -> None:
        """Force application is not implemented in Phase 2C1."""
        raise NotImplementedError("MuJoCoBackend.apply_force() is not implemented in Phase 2C1.")

    def close(self) -> None:
        """Clear loaded MuJoCo objects and ID mappings."""
        self._scene = None
        self._compilation_result = None
        self._model = None
        self._data = None
        self._runtime_body_to_mj_body_id = {}
        self._mj_body_id_to_runtime_body = {}
        self._mj_geom_id_to_runtime_body = {}
        self._runtime_body_to_collision_geom_ids = {}
        self._loaded = False
        self._closed = True
