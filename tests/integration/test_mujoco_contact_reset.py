from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_box, create_ground, create_single_body_asset
from physical_simulation.backends import BackendNotLoadedError, MuJoCoBackend
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _contact_scene():
    ground = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.2, 0.2, 0.2), mass=1.0),
    )
    return create_scene(
        scene_id="contact_reset",
        instances=(
            AssetInstanceSpec("ground_01", ground, fixed_base=True),
            AssetInstanceSpec("box_01", box, Transform(position=(0.0, 0.0, 0.09))),
        ),
    )


def test_reset_contact_determinism() -> None:
    backend = MuJoCoBackend()
    backend.load_scene(_contact_scene())

    first = backend.reset()
    for _ in range(3):
        first = backend.step()

    second = backend.reset()
    for _ in range(3):
        second = backend.step()

    assert len(first.contacts) == len(second.contacts)
    for first_contact, second_contact in zip(first.contacts, second.contacts):
        assert first_contact.body_a == second_contact.body_a
        assert first_contact.body_b == second_contact.body_b
        assert first_contact.position == pytest.approx(second_contact.position, abs=1.0e-12)
        assert first_contact.normal == pytest.approx(second_contact.normal, abs=1.0e-12)
        assert first_contact.penetration_depth == pytest.approx(second_contact.penetration_depth, abs=1.0e-12)


def test_close_then_reload_contacts_work_again() -> None:
    scene = _contact_scene()
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    assert backend.reset().contacts
    backend.close()
    with pytest.raises(BackendNotLoadedError):
        backend.get_contacts()

    backend.load_scene(scene)
    assert backend.reset().contacts
    assert backend.step().contacts
