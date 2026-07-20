from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_single_body_asset, create_sphere
from physical_simulation.backends import BackendNotLoadedError, MuJoCoBackend
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _scene():
    body = create_sphere("sphere_body", 0.1, mass=1.0)
    asset = create_single_body_asset(asset_id="sphere_asset", body=body)
    scene = create_scene(
        scene_id="reset_sphere",
        instances=(AssetInstanceSpec("sphere_01", asset, Transform(position=(0.0, 0.0, 1.0))),),
        gravity=(0.0, 0.0, -9.81),
        timestep=1.0 / 240.0,
    )
    return scene, make_runtime_body_id("sphere_01", "sphere_body")


def test_reset_restores_initial_sphere_state() -> None:
    scene, runtime_id = _scene()
    backend = MuJoCoBackend()
    backend.load_scene(scene)

    initial = backend.reset()
    initial_state = initial.get_body_state(runtime_id)
    assert initial.step_index == 0
    assert initial.time == pytest.approx(0.0)
    assert initial.contacts == ()
    assert initial_state.position == pytest.approx((0.0, 0.0, 1.0))
    assert initial_state.linear_velocity == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-12)

    for _ in range(5):
        backend.step()
    reset = backend.reset()
    reset_state = reset.get_body_state(runtime_id)
    assert reset.step_index == 0
    assert reset.time == pytest.approx(0.0)
    assert reset_state.position == pytest.approx(initial_state.position)
    assert reset_state.linear_velocity == pytest.approx(initial_state.linear_velocity, abs=1.0e-12)
    assert reset.contacts == ()


def test_close_then_reset_step_and_body_state_fail_but_reload_works() -> None:
    scene, runtime_id = _scene()
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    backend.close()

    with pytest.raises(BackendNotLoadedError):
        backend.reset()
    with pytest.raises(BackendNotLoadedError):
        backend.step()
    with pytest.raises(BackendNotLoadedError):
        backend.get_body_state(runtime_id)
    with pytest.raises(BackendNotLoadedError):
        _ = backend.mjcf

    backend.close()
    backend.load_scene(scene)
    assert backend.reset().step_index == 0
    assert backend.step().step_index == 1
