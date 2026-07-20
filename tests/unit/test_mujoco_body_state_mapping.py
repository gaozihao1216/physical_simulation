from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

mujoco = pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_box, create_single_body_asset, create_sphere
from physical_simulation.backends import (
    BackendNotLoadedError,
    MuJoCoBackend,
    MuJoCoRuntimeError,
    UnknownRuntimeBodyError,
    UnsupportedBackendOperationError,
)
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _single_sphere_backend() -> tuple[MuJoCoBackend, str]:
    body = create_sphere("sphere_body", 0.1, mass=1.0)
    asset = create_single_body_asset(asset_id="sphere_asset", body=body)
    scene = create_scene(
        scene_id="unit_body_state",
        instances=(
            AssetInstanceSpec(
                "sphere_01",
                asset,
                Transform(position=(1.0, 2.0, 3.0), rotation=(0.9238795325, 0.0, 0.0, 0.3826834324)),
            ),
        ),
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    return backend, make_runtime_body_id("sphere_01", "sphere_body")


def test_body_state_uses_python_tuples_and_world_pose() -> None:
    backend, runtime_id = _single_sphere_backend()
    state = backend.get_body_state(runtime_id)

    assert state.body_id == runtime_id
    assert isinstance(state.position, tuple)
    assert isinstance(state.rotation, tuple)
    assert isinstance(state.linear_velocity, tuple)
    assert isinstance(state.angular_velocity, tuple)
    assert state.position == pytest.approx((1.0, 2.0, 3.0))
    assert state.rotation == pytest.approx((0.9238795325, 0.0, 0.0, 0.3826834324))
    assert math.sqrt(sum(value * value for value in state.rotation)) == pytest.approx(1.0)


def test_velocity_split_uses_mujoco_rot_then_linear_order(monkeypatch) -> None:
    backend, runtime_id = _single_sphere_backend()

    def fake_object_velocity(_model, _data, _objtype, _objid, result, _flg_local):
        result[:] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    monkeypatch.setattr(mujoco, "mj_objectVelocity", fake_object_velocity)
    state = backend.get_body_state(runtime_id)

    assert state.angular_velocity == pytest.approx((1.0, 2.0, 3.0))
    assert state.linear_velocity == pytest.approx((4.0, 5.0, 6.0))


def test_static_body_velocity_is_near_zero() -> None:
    body = create_box("static_body", (1.0, 1.0, 1.0), body_type="static")
    asset = create_single_body_asset(asset_id="static_asset", body=body)
    scene = create_scene(scene_id="static_velocity", instances=(AssetInstanceSpec("static_01", asset),))
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    state = backend.get_body_state("static_01/static_body")

    assert state.linear_velocity == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-12)
    assert state.angular_velocity == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-12)


def test_unknown_and_unloaded_body_queries_raise_project_errors() -> None:
    backend = MuJoCoBackend()
    with pytest.raises(BackendNotLoadedError):
        backend.get_body_state("missing/body")

    backend, _ = _single_sphere_backend()
    with pytest.raises(UnknownRuntimeBodyError, match="missing/body"):
        backend.get_body_state("missing/body")


def test_non_finite_state_validation_reports_field() -> None:
    backend, _ = _single_sphere_backend()
    state = SimpleNamespace(
        body_id="sphere_01/sphere_body",
        position=(0.0, math.inf, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )

    with pytest.raises(MuJoCoRuntimeError, match="position"):
        backend._validate_finite_state(state)


def test_build_step_result_order_and_empty_joint_contact_state() -> None:
    asset = create_single_body_asset(
        asset_id="sphere_asset",
        body=create_sphere("sphere_body", 0.1, mass=1.0),
    )
    scene = create_scene(
        scene_id="ordered_states",
        instances=(
            AssetInstanceSpec("sphere_01", asset, Transform(position=(0.0, 0.0, 1.0))),
            AssetInstanceSpec("sphere_02", asset, Transform(position=(0.0, 0.0, 2.0))),
        ),
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    result = backend._build_step_result()

    assert [state.body_id for state in result.body_states] == [
        "sphere_01/sphere_body",
        "sphere_02/sphere_body",
    ]
    assert result.joint_states == ()
    assert result.contacts == ()


def test_unsupported_phase_2c2_operations_raise_explicit_errors() -> None:
    backend, runtime_id = _single_sphere_backend()

    with pytest.raises(UnsupportedBackendOperationError):
        backend.step(action={"ignored": False})
    with pytest.raises(UnsupportedBackendOperationError):
        backend.get_joint_state("joint")
    with pytest.raises(UnsupportedBackendOperationError):
        backend.apply_force(runtime_id, (1.0, 0.0, 0.0))
