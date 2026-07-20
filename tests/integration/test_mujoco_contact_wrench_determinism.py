from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_box, create_ground, create_single_body_asset
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _scene():
    ground_asset = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box_asset = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.4, 0.4, 0.4), mass=1.0),
    )
    return create_scene(
        scene_id="contact_wrench_determinism",
        instances=(
            AssetInstanceSpec("ground_01", ground_asset, fixed_base=True),
            AssetInstanceSpec("box_01", box_asset, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=1.0 / 240.0,
    )


def _snapshot(backend: MuJoCoBackend):
    return tuple(
        (
            wrench.contact,
            wrench.force_on_body_a_world,
            wrench.contact_torque_on_body_a_world,
            wrench.force_on_body_b_world,
            wrench.contact_torque_on_body_b_world,
            wrench.normal_force_magnitude,
            wrench.tangential_force_magnitude,
        )
        for wrench in backend.get_contact_wrenches()
    )


def test_contact_wrench_order_and_values_are_deterministic_after_reset() -> None:
    backend = MuJoCoBackend()
    backend.load_scene(_scene())
    backend.reset()
    for _ in range(720):
        backend.step()
    first = _snapshot(backend)

    backend.reset()
    for _ in range(720):
        backend.step()
    second = _snapshot(backend)

    assert len(first) == len(second)
    assert [item[0] for item in first] == [item[0] for item in second]
    for first_item, second_item in zip(first, second):
        for field_index in range(1, 5):
            assert first_item[field_index] == pytest.approx(second_item[field_index], abs=1.0e-10)
        assert first_item[5] == pytest.approx(second_item[5], abs=1.0e-10)
        assert first_item[6] == pytest.approx(second_item[6], abs=1.0e-10)
    backend.close()
