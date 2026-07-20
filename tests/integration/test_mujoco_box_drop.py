from __future__ import annotations

import math

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_box, create_ground, create_single_body_asset
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import SettlingCriteria, evaluate_resting_contact, simulate_body_trajectory
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def test_box_drop_settles_on_ground() -> None:
    ground_asset = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box_asset = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.4, 0.4, 0.4), mass=1.0),
    )
    scene = create_scene(
        scene_id="box_drop_validation",
        instances=(
            AssetInstanceSpec("ground_01", ground_asset, fixed_base=True),
            AssetInstanceSpec("box_01", box_asset, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=1.0 / 240.0,
    )
    runtime_id = make_runtime_body_id("box_01", "box_body")
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    samples = simulate_body_trajectory(backend, runtime_id, steps=720)
    metrics = evaluate_resting_contact(samples, runtime_id)

    assert metrics.initial_height == pytest.approx(1.0)
    assert metrics.contact_step_count > 0
    assert 0.18 <= metrics.final_height <= 0.22
    assert metrics.minimum_height > 0.15
    assert 0.0 <= metrics.maximum_penetration_depth < 0.04
    assert metrics.maximum_linear_speed_last_window <= 0.02
    assert metrics.position_drift_last_window <= 0.002
    assert metrics.final_contact_count > 0
    assert metrics.settled
    assert all(
        math.isfinite(value)
        for sample in samples
        for value in (
            *sample.state.position,
            *sample.state.rotation,
            *sample.state.linear_velocity,
            *sample.state.angular_velocity,
        )
    )
    assert all(
        {contact.body_a, contact.body_b} == {"ground_01/ground_body", "box_01/box_body"}
        for sample in samples
        for contact in sample.contacts
    )
    backend.close()


def test_slightly_tilted_box_damps_and_settles() -> None:
    half_angle = 0.05
    rotation = (math.cos(half_angle), math.sin(half_angle), 0.0, 0.0)
    ground_asset = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box_asset = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.4, 0.4, 0.4), mass=1.0),
    )
    scene = create_scene(
        scene_id="tilted_box_drop",
        instances=(
            AssetInstanceSpec("ground_01", ground_asset, fixed_base=True),
            AssetInstanceSpec("box_01", box_asset, Transform(position=(0.0, 0.0, 1.0), rotation=rotation)),
        ),
        timestep=1.0 / 240.0,
    )
    runtime_id = make_runtime_body_id("box_01", "box_body")
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    samples = simulate_body_trajectory(backend, runtime_id, steps=960)
    metrics = evaluate_resting_contact(samples, runtime_id, criteria=SettlingCriteria(window_steps=120))

    assert metrics.contact_step_count > 0
    assert metrics.maximum_angular_speed > metrics.maximum_angular_speed_last_window
    assert metrics.maximum_angular_speed_last_window <= 0.05
    assert metrics.settled
    backend.close()
