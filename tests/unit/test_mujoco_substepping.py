from __future__ import annotations

from types import SimpleNamespace

import pytest

from physical_simulation.backends import BackendNotLoadedError, MuJoCoBackend
from physical_simulation.mujoco import MuJoCoSubstepRunner, SubstepAdvanceResult
from physical_simulation.mujoco import substepping as substepping_module
from physical_simulation.runtime import SimulationStepResult
from physical_simulation.validation.errors import InvalidRuntimeStateError
from physical_simulation.backends.errors import MuJoCoRuntimeError


def _fake_loaded_backend() -> MuJoCoBackend:
    backend = MuJoCoBackend()
    backend._loaded = True
    backend._model = SimpleNamespace(opt=SimpleNamespace(timestep=0.01))
    backend._data = SimpleNamespace(time=0.0)
    backend._step_index = 0
    return backend


def test_substep_advance_result_validation() -> None:
    result = SimulationStepResult(time=0.0, step_index=0, body_states=())

    advance = SubstepAdvanceResult(1, 8, 1.0 / 240.0, 1.0 / 1920.0, 8, result)

    assert advance.substep_count == 8
    with pytest.raises(MuJoCoRuntimeError, match="substep_count"):
        SubstepAdvanceResult(1, 0, 1.0 / 240.0, 1.0 / 1920.0, 0, result)
    with pytest.raises(InvalidRuntimeStateError):
        SimulationStepResult(time=0.0, step_index=-1, body_states=())


def test_runner_rejects_invalid_macro_timestep_and_unloaded_backend() -> None:
    loaded = _fake_loaded_backend()

    with pytest.raises(MuJoCoRuntimeError, match="macro_timestep"):
        MuJoCoSubstepRunner(loaded, macro_timestep=0.0)
    with pytest.raises(MuJoCoRuntimeError, match="macro_timestep"):
        MuJoCoSubstepRunner(loaded, macro_timestep=float("nan"))
    with pytest.raises(BackendNotLoadedError):
        MuJoCoSubstepRunner(MuJoCoBackend(), macro_timestep=1.0 / 240.0)


def test_runner_rejects_invalid_substep_count() -> None:
    runner = MuJoCoSubstepRunner(_fake_loaded_backend(), macro_timestep=1.0 / 240.0)

    with pytest.raises(MuJoCoRuntimeError, match="substep_count"):
        runner.step(substep_count=0)


def test_runner_restores_timestep_after_internal_mj_step_failure(monkeypatch) -> None:
    backend = _fake_loaded_backend()
    runner = MuJoCoSubstepRunner(backend, macro_timestep=1.0 / 240.0)
    original_timestep = backend._model.opt.timestep

    def raise_step(_model, _data):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        substepping_module,
        "_import_mujoco",
        lambda: SimpleNamespace(mj_step=raise_step, mj_forward=lambda _model, _data: None),
    )

    with pytest.raises(MuJoCoRuntimeError, match="fixed substepping"):
        runner.step(substep_count=4)

    assert backend._model.opt.timestep == original_timestep
    assert runner.macro_step_index == 0
    assert runner.physics_step_count == 0
