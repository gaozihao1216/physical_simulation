"""Trajectory sampling helpers for backend simulation results."""

from __future__ import annotations

from dataclasses import dataclass

from physical_simulation.backends.base import PhysicsBackend
from physical_simulation.runtime import ContactPoint, RigidBodyState


@dataclass(frozen=True)
class BodyStateSample:
    """One sampled body state and contact snapshot."""

    time: float
    step_index: int
    state: RigidBodyState
    contacts: tuple[ContactPoint, ...]


def simulate_body_trajectory(
    backend: PhysicsBackend,
    runtime_body_id: str,
    *,
    steps: int,
) -> tuple[BodyStateSample, ...]:
    """Reset a loaded backend and sample one body for a fixed number of steps."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError(f"steps must be a non-negative int; actual value={steps!r}")

    samples: list[BodyStateSample] = []
    result = backend.reset()
    samples.append(
        BodyStateSample(
            time=result.time,
            step_index=result.step_index,
            state=result.get_body_state(runtime_body_id),
            contacts=result.contacts,
        )
    )
    for _ in range(steps):
        result = backend.step()
        samples.append(
            BodyStateSample(
                time=result.time,
                step_index=result.step_index,
                state=result.get_body_state(runtime_body_id),
                contacts=result.contacts,
            )
        )
    return tuple(samples)
