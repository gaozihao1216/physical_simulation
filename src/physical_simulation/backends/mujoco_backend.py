"""MuJoCo model loading, stepping, and rigid-body state backend."""

from __future__ import annotations

import math
from typing import Any, Optional

from physical_simulation.backends.base import PhysicsBackend
from physical_simulation.backends.errors import (
    BackendNotLoadedError,
    MuJoCoModelLoadingError,
    MuJoCoRuntimeError,
    MuJoCoUnavailableError,
    UnsupportedBackendOperationError,
    UnknownRuntimeBodyError,
    UnknownRuntimeGeomError,
)
from physical_simulation.compilers import MuJoCoCompilationResult, MuJoCoCompiler
from physical_simulation.runtime import ContactPoint, RigidBodyState, SimulationStepResult
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
    """MuJoCo backend for model loading and single-step rigid-body simulation."""

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
        self._runtime_body_order: tuple[str, ...] = ()
        self._step_index = 0
        self._initial_qpos: Any = None
        self._initial_qvel: Any = None
        self._initial_act: Any = None
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
            runtime_body_order = tuple(
                runtime_body_id
                for runtime_body_id, _body_name in compilation_result.runtime_body_to_mujoco_name
            )
            initial_qpos = data.qpos.copy()
            initial_qvel = data.qvel.copy()
            initial_act = data.act.copy()
            self._validate_finite_arrays(
                scene_id=scene.scene_id,
                fields={
                    "qpos": initial_qpos,
                    "qvel": initial_qvel,
                    "act": initial_act,
                },
                step_index=0,
                time=float(data.time),
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
        self._runtime_body_order = runtime_body_order
        self._initial_qpos = initial_qpos
        self._initial_qvel = initial_qvel
        self._initial_act = initial_act
        self._step_index = 0
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
                f"runtime body ID is not loaded; scene_id={self._current_scene_id()!r}, "
                f"runtime_body_id={runtime_body_id!r}"
            )
        return self._runtime_body_to_mj_body_id[runtime_body_id]

    def _require_runtime_geom_id(self, geom_id: int) -> str:
        """Return runtime body id for a private MuJoCo collision geom id."""
        if geom_id not in self._mj_geom_id_to_runtime_body:
            raise UnknownRuntimeGeomError(f"MuJoCo geom ID is not mapped; geom_id={geom_id!r}")
        return self._mj_geom_id_to_runtime_body[geom_id]

    def _current_scene_id(self) -> Optional[str]:
        return self._scene.scene_id if self._scene is not None else None

    def _require_loaded(self, operation: str) -> None:
        if not self._loaded or self._model is None or self._data is None:
            raise BackendNotLoadedError(
                f"{operation} requires a loaded MuJoCo scene; scene_id={self._current_scene_id()!r}"
            )

    def reset(self) -> SimulationStepResult:
        """Reset MuJoCo data to the scene state captured at load_scene()."""
        self._require_loaded("reset")
        mujoco = _import_mujoco()
        try:
            mujoco.mj_resetData(self._model, self._data)
            self._data.qpos[:] = self._initial_qpos
            self._data.qvel[:] = self._initial_qvel
            if self._data.act.size:
                self._data.act[:] = self._initial_act
            if self._data.ctrl.size:
                self._data.ctrl[:] = 0.0
            self._data.qfrc_applied[:] = 0.0
            self._data.xfrc_applied[:] = 0.0
            self._data.qacc_warmstart[:] = 0.0
            mujoco.mj_forward(self._model, self._data)
            self._step_index = 0
            self._validate_finite_backend_state()
            return self._build_step_result()
        except MuJoCoRuntimeError:
            raise
        except Exception as exc:
            raise MuJoCoRuntimeError(
                f"failed to reset MuJoCo backend; scene_id={self._current_scene_id()!r}; "
                f"step_index={self._step_index}; time={self._current_time()}; original_error={exc}"
            ) from exc

    def step(self, action: object | None = None) -> SimulationStepResult:
        """Advance MuJoCo by exactly one physics timestep."""
        self._require_loaded("step")
        if action is not None:
            raise UnsupportedBackendOperationError(
                f"MuJoCoBackend.step() only supports action=None in Phase 2C2; "
                f"scene_id={self._current_scene_id()!r}, step_index={self._step_index}, "
                f"time={self._current_time()}, action={action!r}"
            )
        mujoco = _import_mujoco()
        try:
            mujoco.mj_step(self._model, self._data)
            self._step_index += 1
            mujoco.mj_forward(self._model, self._data)
            self._validate_finite_backend_state()
            return self._build_step_result()
        except MuJoCoRuntimeError:
            raise
        except Exception as exc:
            raise MuJoCoRuntimeError(
                f"failed to step MuJoCo backend; scene_id={self._current_scene_id()!r}; "
                f"step_index={self._step_index}; time={self._current_time()}; original_error={exc}"
            ) from exc

    def get_body_state(self, runtime_body_id: str) -> RigidBodyState:
        """Return world-space rigid-body state for a runtime body id."""
        self._require_loaded("get_body_state")
        mj_body_id = self._require_runtime_body_id(runtime_body_id)
        try:
            position = self._float_tuple(self._data.xpos[mj_body_id], 3)
            rotation = self._float_tuple(self._data.xquat[mj_body_id], 4)
            linear_velocity, angular_velocity = self._read_body_world_velocity(mj_body_id)
            state = RigidBodyState(
                body_id=runtime_body_id,
                position=position,
                rotation=rotation,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
            )
            self._validate_finite_state(state)
            return state
        except (MuJoCoRuntimeError, UnknownRuntimeBodyError):
            raise
        except Exception as exc:
            raise MuJoCoRuntimeError(
                f"failed to read MuJoCo body state; scene_id={self._current_scene_id()!r}; "
                f"runtime_body_id={runtime_body_id!r}; step_index={self._step_index}; "
                f"time={self._current_time()}; original_error={exc}"
            ) from exc

    def get_joint_state(self, joint_id: str) -> Any:
        """Joint state extraction is not implemented in Phase 2C2."""
        raise UnsupportedBackendOperationError(
            f"MuJoCoBackend.get_joint_state() is not implemented in Phase 2C2; "
            f"scene_id={self._current_scene_id()!r}, joint_id={joint_id!r}"
        )

    def get_contacts(self) -> tuple[ContactPoint, ...]:
        """Return contacts.

        Contact extraction is implemented in Phase 2D. Phase 2C2 always
        returns an empty tuple even if MuJoCo has internal contacts.
        """
        self._require_loaded("get_contacts")
        return ()

    def apply_force(self, body_id: str, force: Any, point: Optional[Any] = None) -> None:
        """Force application is not implemented in Phase 2C2."""
        raise UnsupportedBackendOperationError(
            f"MuJoCoBackend.apply_force() is not implemented in Phase 2C2; "
            f"scene_id={self._current_scene_id()!r}, body_id={body_id!r}, "
            f"step_index={self._step_index}, time={self._current_time()}"
        )

    def _read_body_world_velocity(
        self,
        mj_body_id: int,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return linear velocity and angular velocity in world coordinates."""
        mujoco = _import_mujoco()
        import numpy as np

        velocity = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self._model,
            self._data,
            mujoco.mjtObj.mjOBJ_BODY,
            mj_body_id,
            velocity,
            0,
        )
        angular_velocity = self._float_tuple(velocity[:3], 3)
        linear_velocity = self._float_tuple(velocity[3:], 3)
        return linear_velocity, angular_velocity

    def _build_step_result(self) -> SimulationStepResult:
        """Build a backend-independent snapshot from current MuJoCo data."""
        self._require_loaded("_build_step_result")
        states = tuple(self.get_body_state(runtime_body_id) for runtime_body_id in self._runtime_body_order)
        result = SimulationStepResult(
            time=float(self._data.time),
            step_index=self._step_index,
            body_states=states,
            joint_states=(),
            contacts=(),
        )
        return result

    def _validate_finite_backend_state(self) -> None:
        self._validate_finite_arrays(
            scene_id=self._current_scene_id(),
            fields={
                "time": (float(self._data.time),),
                "qpos": self._data.qpos,
                "qvel": self._data.qvel,
                "act": self._data.act,
            },
            step_index=self._step_index,
            time=self._current_time(),
        )

    def _validate_finite_arrays(
        self,
        *,
        scene_id: Optional[str],
        fields: dict[str, Any],
        step_index: int,
        time: float,
    ) -> None:
        for field_name, values in fields.items():
            for value in values:
                if not math.isfinite(float(value)):
                    raise MuJoCoRuntimeError(
                        f"non-finite MuJoCo backend state; scene_id={scene_id!r}; "
                        f"step_index={step_index}; time={time}; field={field_name!r}; value={value!r}"
                    )

    def _validate_finite_state(self, state: RigidBodyState) -> None:
        fields = {
            "position": state.position,
            "rotation": state.rotation,
            "linear_velocity": state.linear_velocity,
            "angular_velocity": state.angular_velocity,
        }
        for field_name, values in fields.items():
            for value in values:
                if not math.isfinite(value):
                    raise MuJoCoRuntimeError(
                        f"non-finite rigid body state; scene_id={self._current_scene_id()!r}; "
                        f"runtime_body_id={state.body_id!r}; step_index={self._step_index}; "
                        f"time={self._current_time()}; field={field_name!r}; value={value!r}"
                    )

    def _current_time(self) -> float:
        if self._data is None:
            return 0.0
        return float(self._data.time)

    def _float_tuple(self, values: Any, length: int) -> tuple[float, ...]:
        result = tuple(float(value) for value in values)
        if len(result) != length:
            raise MuJoCoRuntimeError(
                f"unexpected MuJoCo vector length; scene_id={self._current_scene_id()!r}; "
                f"step_index={self._step_index}; time={self._current_time()}; "
                f"expected_length={length}; actual_length={len(result)}"
            )
        return result

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
        self._runtime_body_order = ()
        self._step_index = 0
        self._initial_qpos = None
        self._initial_qvel = None
        self._initial_act = None
        self._loaded = False
        self._closed = True
