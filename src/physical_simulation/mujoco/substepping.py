"""Fixed substepping utilities for MuJoCo backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from physical_simulation.backends.errors import BackendNotLoadedError, MuJoCoRuntimeError
from physical_simulation.backends.mujoco_backend import MuJoCoBackend, _import_mujoco
from physical_simulation.runtime import SimulationStepResult
from physical_simulation.validation.asset_validator import _finite_float


@dataclass(frozen=True)
class SubstepAdvanceResult:
    """Result for one externally visible macro step advanced with fixed substeps."""

    macro_step_index: int
    physics_step_count: int
    macro_timestep: float
    substep_timestep: float
    substep_count: int
    simulation_result: SimulationStepResult

    def __post_init__(self) -> None:
        if not isinstance(self.macro_step_index, int) or isinstance(self.macro_step_index, bool) or self.macro_step_index < 0:
            raise MuJoCoRuntimeError(f"macro_step_index must be a non-negative int; actual value={self.macro_step_index!r}")
        if not isinstance(self.physics_step_count, int) or isinstance(self.physics_step_count, bool) or self.physics_step_count < 0:
            raise MuJoCoRuntimeError(f"physics_step_count must be a non-negative int; actual value={self.physics_step_count!r}")
        object.__setattr__(
            self,
            "macro_timestep",
            _finite_float(self.macro_timestep, field_name="macro_timestep", minimum=0.0, strict_minimum=True, error_type=MuJoCoRuntimeError),
        )
        object.__setattr__(
            self,
            "substep_timestep",
            _finite_float(self.substep_timestep, field_name="substep_timestep", minimum=0.0, strict_minimum=True, error_type=MuJoCoRuntimeError),
        )
        if not isinstance(self.substep_count, int) or isinstance(self.substep_count, bool) or self.substep_count < 1:
            raise MuJoCoRuntimeError(f"substep_count must be an int >= 1; actual value={self.substep_count!r}")
        if not isinstance(self.simulation_result, SimulationStepResult):
            raise MuJoCoRuntimeError(f"simulation_result must be SimulationStepResult; actual value={self.simulation_result!r}")


class MuJoCoSubstepRunner:
    """Advance a loaded MuJoCoBackend with a fixed number of internal substeps."""

    def __init__(self, backend: MuJoCoBackend, *, macro_timestep: float) -> None:
        if not isinstance(backend, MuJoCoBackend):
            raise MuJoCoRuntimeError(f"backend must be MuJoCoBackend; actual value={backend!r}")
        self._backend = backend
        self._macro_timestep = _finite_float(
            macro_timestep,
            field_name="macro_timestep",
            minimum=0.0,
            strict_minimum=True,
            error_type=MuJoCoRuntimeError,
        )
        self._macro_step_index = 0
        self._physics_step_count = 0
        self._require_loaded("MuJoCoSubstepRunner")

    @property
    def macro_step_index(self) -> int:
        """Return the number of completed externally visible macro steps."""
        return self._macro_step_index

    @property
    def physics_step_count(self) -> int:
        """Return the cumulative number of internal mj_step calls run by this runner."""
        return self._physics_step_count

    @property
    def macro_timestep(self) -> float:
        """Return the externally visible timestep advanced by each runner step."""
        return self._macro_timestep

    def reset(self) -> SimulationStepResult:
        """Reset the wrapped backend and clear runner counters."""
        result = self._backend.reset()
        self._macro_step_index = 0
        self._physics_step_count = 0
        return result

    def step(
        self,
        *,
        substep_count: int,
        substep_callback: Callable[[SimulationStepResult], None] | None = None,
    ) -> SubstepAdvanceResult:
        """Advance one macro timestep using ``substep_count`` MuJoCo physics steps."""
        self._require_loaded("MuJoCoSubstepRunner.step")
        if not isinstance(substep_count, int) or isinstance(substep_count, bool) or substep_count < 1:
            raise MuJoCoRuntimeError(f"substep_count must be an int >= 1; actual value={substep_count!r}")
        substep_timestep = self._macro_timestep / substep_count
        if not math.isfinite(substep_timestep) or substep_timestep <= 0.0:
            raise MuJoCoRuntimeError(
                f"substep_timestep must be finite and > 0; macro_timestep={self._macro_timestep!r}, "
                f"substep_count={substep_count!r}, substep_timestep={substep_timestep!r}"
            )

        model = self._backend._model
        data = self._backend._data
        original_timestep = float(model.opt.timestep)
        start_time = float(data.time)
        completed_substeps = 0
        mujoco = _import_mujoco()
        try:
            model.opt.timestep = substep_timestep
            for _ in range(substep_count):
                mujoco.mj_step(model, data)
                completed_substeps += 1
                self._backend._step_index += 1
                if substep_callback is not None:
                    mujoco.mj_forward(model, data)
                    substep_callback(self._backend._build_step_result())
            mujoco.mj_forward(model, data)
        except Exception as exc:
            raise MuJoCoRuntimeError(
                f"failed during MuJoCo fixed substepping; scene_id={self._backend._current_scene_id()!r}; "
                f"macro_step_index={self._macro_step_index}; completed_substeps={completed_substeps}; "
                f"substep_count={substep_count}; original_error={exc}"
            ) from exc
        finally:
            model.opt.timestep = original_timestep

        self._backend._validate_finite_backend_state()
        elapsed = float(data.time) - start_time
        if not math.isclose(elapsed, self._macro_timestep, rel_tol=1.0e-10, abs_tol=1.0e-12):
            raise MuJoCoRuntimeError(
                f"substepping advanced unexpected simulation time; expected={self._macro_timestep!r}, actual={elapsed!r}, "
                f"substep_count={substep_count!r}"
            )
        self._macro_step_index += 1
        self._physics_step_count += completed_substeps
        return SubstepAdvanceResult(
            macro_step_index=self._macro_step_index,
            physics_step_count=self._physics_step_count,
            macro_timestep=self._macro_timestep,
            substep_timestep=substep_timestep,
            substep_count=substep_count,
            simulation_result=self._backend._build_step_result(),
        )

    def _require_loaded(self, operation: str) -> None:
        if not self._backend.is_loaded or self._backend._model is None or self._backend._data is None:
            raise BackendNotLoadedError(
                f"{operation} requires a loaded MuJoCoBackend; scene_id={self._backend._current_scene_id()!r}"
            )
