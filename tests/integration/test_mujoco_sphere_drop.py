from __future__ import annotations

import math

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_ground, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import evaluate_resting_contact, simulate_body_trajectory
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def test_sphere_drop_settles_on_ground() -> None:
    ground_asset = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    sphere_asset = create_single_body_asset(
        asset_id="sphere_asset",
        body=create_sphere("sphere_body", 0.1, mass=1.0),
    )
    scene = create_scene(
        scene_id="sphere_drop_validation",
        instances=(
            AssetInstanceSpec("ground_01", ground_asset, fixed_base=True),
            AssetInstanceSpec("sphere_01", sphere_asset, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=1.0 / 240.0,
    )
    runtime_id = make_runtime_body_id("sphere_01", "sphere_body")
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    samples = simulate_body_trajectory(backend, runtime_id, steps=720)
    metrics = evaluate_resting_contact(samples, runtime_id)

    assert metrics.contact_step_count > 0
    assert 0.09 <= metrics.final_height <= 0.11
    assert metrics.minimum_height > 0.06
    assert 0.0 <= metrics.maximum_penetration_depth < 0.04
    assert metrics.maximum_linear_speed_last_window <= 0.02
    assert metrics.maximum_angular_speed_last_window <= 0.05
    assert metrics.final_contact_count > 0
    assert metrics.settled
    assert all(
        math.isfinite(value)
        for sample in samples
        for contact in sample.contacts
        for value in (*contact.position, *contact.normal, contact.penetration_depth)
    )
    backend.close()
