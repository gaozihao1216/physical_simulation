from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import (
    BoxGeometry,
    ColliderSpec,
    RigidBodySpec,
    Transform,
    create_box,
    create_single_body_asset,
    create_sphere,
)
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.evaluation import evaluate_resting_contact, simulate_body_trajectory
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def test_dynamic_sphere_settles_on_compound_table_surface() -> None:
    table_body = RigidBodySpec(
        "table_body",
        "table_body",
        "static",
        Transform.identity(),
        (),
        (
            ColliderSpec("top", BoxGeometry((1.2, 1.2, 0.08)), Transform(position=(0.0, 0.0, 0.5))),
            ColliderSpec("leg_1", BoxGeometry((0.08, 0.08, 0.5)), Transform(position=(-0.45, -0.45, 0.25))),
            ColliderSpec("leg_2", BoxGeometry((0.08, 0.08, 0.5)), Transform(position=(0.45, -0.45, 0.25))),
            ColliderSpec("leg_3", BoxGeometry((0.08, 0.08, 0.5)), Transform(position=(-0.45, 0.45, 0.25))),
            ColliderSpec("leg_4", BoxGeometry((0.08, 0.08, 0.5)), Transform(position=(0.45, 0.45, 0.25))),
        ),
    )
    table_asset = create_single_body_asset(asset_id="table_asset", body=table_body)
    sphere_asset = create_single_body_asset(
        asset_id="sphere_asset",
        body=create_sphere("sphere_body", 0.1, mass=1.0),
    )
    scene = create_scene(
        scene_id="compound_surface",
        instances=(
            AssetInstanceSpec("table_01", table_asset, fixed_base=True),
            AssetInstanceSpec("sphere_01", sphere_asset, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=1.0 / 240.0,
    )
    runtime_id = make_runtime_body_id("sphere_01", "sphere_body")
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    assert len(backend._runtime_body_to_collision_geom_ids["table_01/table_body"]) == 5

    samples = simulate_body_trajectory(backend, runtime_id, steps=720)
    metrics = evaluate_resting_contact(samples, runtime_id)

    assert 0.55 <= metrics.final_height <= 0.65
    assert metrics.minimum_height > 0.5
    assert metrics.final_contact_count > 0
    assert metrics.settled
    assert all(
        "table_01/table_body" in {contact.body_a, contact.body_b}
        for contact in samples[-1].contacts
        if runtime_id in {contact.body_a, contact.body_b}
    )
    assert all(contact.body_a != contact.body_b for sample in samples for contact in sample.contacts)
    backend.close()
