from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_box, create_ground, create_single_body_asset
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import evaluate_resting_contact, simulate_body_trajectory
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def test_box_drop_replay_is_deterministic() -> None:
    ground_asset = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box_asset = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.4, 0.4, 0.4), mass=1.0),
    )
    scene = create_scene(
        scene_id="drop_determinism",
        instances=(
            AssetInstanceSpec("ground_01", ground_asset, fixed_base=True),
            AssetInstanceSpec("box_01", box_asset, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=1.0 / 240.0,
    )
    runtime_id = make_runtime_body_id("box_01", "box_body")
    backend = MuJoCoBackend()
    backend.load_scene(scene)

    first = simulate_body_trajectory(backend, runtime_id, steps=360)
    second = simulate_body_trajectory(backend, runtime_id, steps=360)
    first_metrics = evaluate_resting_contact(first, runtime_id)
    second_metrics = evaluate_resting_contact(second, runtime_id)

    assert len(first) == len(second)
    assert [len(sample.contacts) for sample in first] == [len(sample.contacts) for sample in second]
    for first_contact, second_contact in zip(first[-1].contacts, second[-1].contacts):
        assert first_contact.body_a == second_contact.body_a
        assert first_contact.body_b == second_contact.body_b
        assert first_contact.position == pytest.approx(second_contact.position)
        assert first_contact.normal == pytest.approx(second_contact.normal)
        assert first_contact.penetration_depth == pytest.approx(second_contact.penetration_depth)
    assert first[-1].state.position == pytest.approx(second[-1].state.position)
    assert first[-1].state.rotation == pytest.approx(second[-1].state.rotation)
    assert first[-1].state.linear_velocity == pytest.approx(second[-1].state.linear_velocity)
    assert first_metrics.final_height == pytest.approx(second_metrics.final_height)
    assert first_metrics.maximum_penetration_depth == pytest.approx(second_metrics.maximum_penetration_depth)
    assert first_metrics.settled == second_metrics.settled
    backend.close()
