from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _run_steps(backend: MuJoCoBackend, count: int):
    backend.reset()
    result = None
    for _ in range(count):
        result = backend.step()
    assert result is not None
    return result


def test_reset_makes_repeated_runs_deterministic() -> None:
    asset = create_single_body_asset(
        asset_id="sphere_asset",
        body=create_sphere("sphere_body", 0.1, mass=1.0),
    )
    scene = create_scene(
        scene_id="reset_determinism",
        instances=(AssetInstanceSpec("sphere_01", asset, Transform(position=(0.0, 0.0, 1.0))),),
        gravity=(0.0, 0.0, -9.81),
        timestep=1.0 / 240.0,
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)

    first = _run_steps(backend, 60)
    second = _run_steps(backend, 60)

    assert first.step_index == second.step_index == 60
    assert first.time == pytest.approx(second.time)
    assert [state.body_id for state in first.body_states] == [state.body_id for state in second.body_states]
    assert first.contacts == second.contacts == ()
    for first_state, second_state in zip(first.body_states, second.body_states):
        assert first_state.position == pytest.approx(second_state.position, abs=1.0e-12)
        assert first_state.rotation == pytest.approx(second_state.rotation, abs=1.0e-12)
        assert first_state.linear_velocity == pytest.approx(second_state.linear_velocity, abs=1.0e-12)
        assert first_state.angular_velocity == pytest.approx(second_state.angular_velocity, abs=1.0e-12)
