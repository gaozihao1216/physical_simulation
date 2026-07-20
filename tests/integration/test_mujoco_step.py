from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def test_single_step_advances_one_timestep() -> None:
    body = create_sphere("sphere_body", 0.1, mass=1.0)
    asset = create_single_body_asset(asset_id="sphere_asset", body=body)
    scene = create_scene(
        scene_id="single_step",
        instances=(AssetInstanceSpec("sphere_01", asset, Transform(position=(0.0, 0.0, 1.0))),),
        gravity=(0.0, 0.0, -9.81),
        timestep=1.0 / 240.0,
    )
    runtime_id = make_runtime_body_id("sphere_01", "sphere_body")
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    initial = backend.reset().get_body_state(runtime_id)

    first = backend.step()
    first_state = first.get_body_state(runtime_id)
    assert first.step_index == 1
    assert first.time == pytest.approx(scene.timestep)
    assert first_state.position[2] < initial.position[2]
    assert first_state.linear_velocity[2] < 0.0

    for _ in range(9):
        result = backend.step()
    assert result.step_index == 10
    assert result.time == pytest.approx(10 * scene.timestep)
    assert result.contacts == ()
