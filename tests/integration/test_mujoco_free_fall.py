from __future__ import annotations

import math

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def test_free_fall_trend_without_contact_extraction() -> None:
    body = create_sphere("sphere_body", 0.1, mass=1.0)
    asset = create_single_body_asset(asset_id="sphere_asset", body=body)
    scene = create_scene(
        scene_id="free_fall",
        instances=(AssetInstanceSpec("sphere_01", asset, Transform(position=(0.0, 0.0, 1.0))),),
        gravity=(0.0, 0.0, -9.81),
        timestep=1.0 / 240.0,
    )
    runtime_id = make_runtime_body_id("sphere_01", "sphere_body")
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    initial_state = backend.reset().get_body_state(runtime_id)

    result = None
    for _ in range(60):
        result = backend.step()
        assert result.contacts == ()

    assert result is not None
    state = result.get_body_state(runtime_id)
    t = result.time
    assert state.position[2] < initial_state.position[2]
    assert state.linear_velocity[2] < 0.0
    assert state.position[0] == pytest.approx(0.0, abs=1.0e-10)
    assert state.position[1] == pytest.approx(0.0, abs=1.0e-10)
    assert state.linear_velocity[0] == pytest.approx(0.0, abs=1.0e-10)
    assert state.linear_velocity[1] == pytest.approx(0.0, abs=1.0e-10)
    assert state.angular_velocity == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-10)
    assert state.position[2] == pytest.approx(1.0 + 0.5 * scene.gravity[2] * t * t, abs=0.02)
    assert state.linear_velocity[2] == pytest.approx(scene.gravity[2] * t, abs=0.05)
    assert all(
        math.isfinite(value)
        for value in (
            *state.position,
            *state.rotation,
            *state.linear_velocity,
            *state.angular_velocity,
        )
    )
