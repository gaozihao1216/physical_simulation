from __future__ import annotations

import pytest

from physical_simulation.evaluation import simulate_body_trajectory
from physical_simulation.runtime import RigidBodyState, SimulationStepResult


def _state(body_id: str, step: int = 0) -> RigidBodyState:
    return RigidBodyState(
        body_id=body_id,
        position=(0.0, 0.0, float(step)),
        rotation=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )


class _FakeBackend:
    def __init__(self) -> None:
        self.index = 0

    def reset(self) -> SimulationStepResult:
        self.index = 0
        return SimulationStepResult(0.0, 0, (_state("body", 0),))

    def step(self, action=None) -> SimulationStepResult:
        self.index += 1
        return SimulationStepResult(float(self.index), self.index, (_state("body", self.index),))


def test_steps_zero_returns_reset_sample() -> None:
    samples = simulate_body_trajectory(_FakeBackend(), "body", steps=0)
    assert len(samples) == 1
    assert samples[0].step_index == 0
    assert samples[0].time == 0.0


def test_steps_n_returns_n_plus_one_samples_with_monotonic_time() -> None:
    samples = simulate_body_trajectory(_FakeBackend(), "body", steps=3)
    assert len(samples) == 4
    assert [sample.step_index for sample in samples] == [0, 1, 2, 3]
    assert [sample.time for sample in samples] == sorted(sample.time for sample in samples)


def test_negative_steps_raise() -> None:
    with pytest.raises(ValueError, match="steps"):
        simulate_body_trajectory(_FakeBackend(), "body", steps=-1)


def test_unknown_runtime_body_raises_from_step_result_lookup() -> None:
    with pytest.raises(KeyError, match="missing"):
        simulate_body_trajectory(_FakeBackend(), "missing", steps=0)
