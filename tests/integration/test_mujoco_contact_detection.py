from __future__ import annotations

import math

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_box, create_ground, create_single_body_asset
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.scene import AssetInstanceSpec, create_scene


def test_initial_overlap_contact_maps_to_runtime_body_ids() -> None:
    ground = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.2, 0.2, 0.2), mass=1.0),
    )
    scene = create_scene(
        scene_id="initial_overlap_contact",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("box_01", box, Transform(position=(0.0, 0.0, 0.09)), fixed_base=True),
        ),
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    result = backend.reset()

    assert result.contacts
    assert result.contacts == backend.get_contacts()
    for contact in result.contacts:
        assert {contact.body_a, contact.body_b} == {"ground_01/ground_body", "box_01/box_body"}
        assert contact.body_a != contact.body_b
        assert contact.penetration_depth >= 0.0
        assert all(math.isfinite(value) for value in (*contact.position, *contact.normal, contact.penetration_depth))
        assert contact.normal_force is None
        assert contact.tangential_force is None


def test_dynamic_body_eventually_produces_contact_with_ground() -> None:
    ground = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.2, 0.2, 0.2), mass=1.0),
    )
    scene = create_scene(
        scene_id="dynamic_contact",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("box_01", box, Transform(position=(0.0, 0.0, 0.5))),
        ),
        timestep=1.0 / 240.0,
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    assert backend.reset().contacts == ()

    result = None
    for _ in range(120):
        result = backend.step()
        if result.contacts:
            break

    assert result is not None
    assert result.contacts
    assert result.contacts == backend.get_contacts()
    assert all({contact.body_a, contact.body_b} == {"ground_01/ground_body", "box_01/box_body"} for contact in result.contacts)
