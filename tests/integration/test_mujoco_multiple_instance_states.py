from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def test_multiple_sphere_instances_keep_distinct_states() -> None:
    asset = create_single_body_asset(
        asset_id="sphere_asset",
        body=create_sphere("sphere_body", 0.1, mass=1.0),
    )
    scene = create_scene(
        scene_id="multiple_spheres",
        instances=(
            AssetInstanceSpec("sphere_01", asset, Transform(position=(0.0, 0.0, 1.0))),
            AssetInstanceSpec("sphere_02", asset, Transform(position=(0.0, 0.0, 2.0))),
        ),
        gravity=(0.0, 0.0, -9.81),
        timestep=1.0 / 240.0,
    )
    first_id = make_runtime_body_id("sphere_01", "sphere_body")
    second_id = make_runtime_body_id("sphere_02", "sphere_body")
    backend = MuJoCoBackend()
    backend.load_scene(scene)

    initial = backend.reset()
    assert [state.body_id for state in initial.body_states] == [first_id, second_id]
    assert initial.get_body_state(first_id).position[2] == pytest.approx(1.0)
    assert initial.get_body_state(second_id).position[2] == pytest.approx(2.0)

    for _ in range(5):
        result = backend.step()
    first = backend.get_body_state(first_id)
    second = result.get_body_state(second_id)

    assert first.position[2] < 1.0
    assert second.position[2] < 2.0
    assert second.position[2] > first.position[2]
    assert first.linear_velocity[2] < 0.0
    assert second.linear_velocity[2] < 0.0
    assert result.get_body_state(first_id).body_id == first_id
    assert result.get_body_state(second_id).body_id == second_id
