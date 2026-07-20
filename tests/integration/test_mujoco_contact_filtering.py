from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import BoxGeometry, ColliderSpec, RigidBodySpec, Transform, create_box, create_ground, create_single_body_asset
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.scene import AssetInstanceSpec, create_scene


def test_compound_collider_external_contacts_map_to_one_runtime_body() -> None:
    ground = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    foot = BoxGeometry((0.2, 0.2, 0.2))
    compound_body = RigidBodySpec(
        "compound_body",
        "compound_body",
        "dynamic",
        Transform.identity(),
        (),
        (
            ColliderSpec("left", foot, Transform(position=(-0.15, 0.0, 0.0))),
            ColliderSpec("right", foot, Transform(position=(0.15, 0.0, 0.0))),
        ),
        mass_properties=create_box("tmp", (0.4, 0.2, 0.2), mass=1.0).mass_properties,
    )
    compound = create_single_body_asset(asset_id="compound_asset", body=compound_body)
    scene = create_scene(
        scene_id="compound_contacts",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("compound_01", compound, Transform(position=(0.0, 0.0, 0.09))),
        ),
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    contacts = backend.reset().contacts

    assert contacts
    assert all(contact.body_a != contact.body_b for contact in contacts)
    assert all(
        {contact.body_a, contact.body_b} == {"ground_01/ground_body", "compound_01/compound_body"}
        for contact in contacts
    )
