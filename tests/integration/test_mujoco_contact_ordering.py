from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_box, create_ground, create_single_body_asset
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _box_on_ground_contacts():
    ground = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.2, 0.2, 0.2), mass=1.0),
    )
    scene = create_scene(
        scene_id="contact_ordering",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("box_01", box, Transform(position=(0.0, 0.0, 0.09))),
        ),
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    return backend.reset().contacts


def test_multiple_contact_points_are_preserved_and_sorted() -> None:
    contacts = _box_on_ground_contacts()
    repeated = _box_on_ground_contacts()

    assert len(contacts) >= 2
    assert len({contact.position for contact in contacts}) >= 2
    assert contacts == tuple(sorted(contacts, key=MuJoCoBackend()._contact_sort_key))
    assert len(contacts) == len(repeated)
    assert [contact.body_a for contact in contacts] == [contact.body_a for contact in repeated]
    assert [contact.body_b for contact in contacts] == [contact.body_b for contact in repeated]
    for first, second in zip(contacts, repeated):
        assert first.position == pytest.approx(second.position)
        assert first.normal == pytest.approx(second.normal)
        assert first.penetration_depth == pytest.approx(second.penetration_depth)
