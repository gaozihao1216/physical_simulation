import pytest

from physical_simulation.assets import Transform, create_box, create_single_body_asset
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _free_box_backend(gravity=(0.0, 0.0, -9.81)) -> tuple[MuJoCoBackend, str]:
    body = create_box("box_body", (0.2, 0.2, 0.2), mass=1.0)
    asset = create_single_body_asset(asset_id="box_asset", body=body)
    scene = create_scene(
        scene_id="control_box",
        instances=(AssetInstanceSpec("box_01", asset, Transform(position=(0.0, 0.0, 1.0))),),
        gravity=gravity,
        timestep=1.0 / 240.0,
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    return backend, make_runtime_body_id("box_01", "box_body")


def test_set_body_velocity_changes_motion_after_step() -> None:
    backend, runtime_id = _free_box_backend(gravity=(0.0, 0.0, 0.0))
    initial = backend.reset().get_body_state(runtime_id)

    backend.set_body_velocity(runtime_id, (1.0, 0.0, 0.0))
    state = backend.step().get_body_state(runtime_id)

    assert state.position[0] > initial.position[0]
    assert state.linear_velocity[0] == pytest.approx(1.0)


def test_update_initial_velocity_survives_reset_and_moves_after_step() -> None:
    backend, runtime_id = _free_box_backend(gravity=(0.0, 0.0, 0.0))

    backend.set_body_velocity(runtime_id, (0.0, 1.0, 0.0), update_initial=True)
    reset_state = backend.reset().get_body_state(runtime_id)
    stepped_state = backend.step().get_body_state(runtime_id)

    assert reset_state.linear_velocity == pytest.approx((0.0, 1.0, 0.0))
    assert stepped_state.position[1] > reset_state.position[1]


def test_apply_force_changes_linear_velocity_and_clear_stops_acceleration() -> None:
    backend, runtime_id = _free_box_backend(gravity=(0.0, 0.0, 0.0))
    backend.reset()

    backend.apply_force(runtime_id, (10.0, 0.0, 0.0))
    first = backend.step().get_body_state(runtime_id)
    backend.clear_applied_forces()
    second = backend.step().get_body_state(runtime_id)

    assert first.linear_velocity[0] > 0.0
    assert second.linear_velocity[0] == pytest.approx(first.linear_velocity[0])


def test_apply_force_at_offset_point_creates_angular_velocity() -> None:
    backend, runtime_id = _free_box_backend(gravity=(0.0, 0.0, 0.0))
    backend.reset()

    backend.apply_force(runtime_id, (10.0, 0.0, 0.0), point=(0.0, 0.1, 1.0))
    state = backend.step().get_body_state(runtime_id)

    assert state.linear_velocity[0] > 0.0
    assert abs(state.angular_velocity[2]) > 0.0


def test_apply_torque_changes_angular_velocity_without_linear_force() -> None:
    backend, runtime_id = _free_box_backend(gravity=(0.0, 0.0, 0.0))
    backend.reset()

    backend.apply_torque(runtime_id, (0.0, 0.0, 1.0))
    state = backend.step().get_body_state(runtime_id)

    assert state.linear_velocity == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-12)
    assert abs(state.angular_velocity[2]) > 0.0
