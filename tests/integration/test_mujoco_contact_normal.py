from __future__ import annotations

import math

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_box, create_ground, create_single_body_asset
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.scene import AssetInstanceSpec, create_scene


def test_contact_normal_points_from_body_a_to_body_b() -> None:
    ground = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.2, 0.2, 0.2), mass=1.0),
    )
    scene = create_scene(
        scene_id="contact_normal",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("box_01", box, Transform(position=(0.0, 0.0, 0.09))),
        ),
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    contacts = backend.reset().contacts

    assert contacts
    for contact in contacts:
        if contact.body_a == "box_01/box_body" and contact.body_b == "ground_01/ground_body":
            assert contact.normal[2] < -0.9
        elif contact.body_a == "ground_01/ground_body" and contact.body_b == "box_01/box_body":
            assert contact.normal[2] > 0.9
        else:
            raise AssertionError(f"unexpected contact pair: {contact!r}")
        assert math.sqrt(sum(value * value for value in contact.normal)) == pytest.approx(1.0)
