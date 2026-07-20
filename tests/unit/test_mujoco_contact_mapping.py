from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import (
    BoxGeometry,
    ColliderSpec,
    RigidBodySpec,
    Transform,
    VisualSpec,
    create_box,
    create_single_body_asset,
)
from physical_simulation.backends import MuJoCoBackend, MuJoCoRuntimeError
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _fake_contact(geom1: int, geom2: int, *, dist: float = -0.25, normal=(0.0, 0.0, 2.0), pos=(1.0, 2.0, 3.0)):
    return SimpleNamespace(
        geom1=geom1,
        geom2=geom2,
        dist=dist,
        pos=pos,
        frame=tuple(normal) + (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )


def _backend_with_fake_geom_mapping() -> MuJoCoBackend:
    backend = MuJoCoBackend()
    backend._loaded = True
    backend._model = SimpleNamespace(ngeom=3)
    backend._data = SimpleNamespace(time=0.0, ncon=0, contact=())
    backend._scene = SimpleNamespace(scene_id="unit_contacts")
    backend._mj_geom_id_to_runtime_body = {
        1: "box_01/box_body",
        2: "ground_01/ground_body",
        3: "box_01/box_body",
    }
    return backend


def test_geom_numeric_id_maps_to_runtime_body_id() -> None:
    backend = _backend_with_fake_geom_mapping()
    assert backend._get_runtime_body_for_geom_id(1) == "box_01/box_body"


def test_unknown_collision_geom_raises_runtime_error() -> None:
    backend = _backend_with_fake_geom_mapping()
    with pytest.raises(MuJoCoRuntimeError, match="geom_id=99"):
        backend._get_runtime_body_for_geom_id(99)


def test_same_runtime_body_contact_is_filtered() -> None:
    backend = _backend_with_fake_geom_mapping()
    assert backend._convert_mujoco_contact(_fake_contact(1, 3)) is None


def test_body_ordering_flips_normal_when_sorted_order_swaps() -> None:
    backend = _backend_with_fake_geom_mapping()
    contact = backend._convert_mujoco_contact(_fake_contact(2, 1, normal=(0.0, 0.0, 5.0)))

    assert contact is not None
    assert contact.body_a == "box_01/box_body"
    assert contact.body_b == "ground_01/ground_body"
    assert contact.normal == pytest.approx((0.0, 0.0, -1.0))


def test_normal_is_normalized_and_forces_are_not_filled() -> None:
    backend = _backend_with_fake_geom_mapping()
    contact = backend._convert_mujoco_contact(_fake_contact(1, 2, normal=(0.0, 0.0, 3.0)))

    assert contact is not None
    assert contact.normal == pytest.approx((0.0, 0.0, 1.0))
    assert math.sqrt(sum(value * value for value in contact.normal)) == pytest.approx(1.0)
    assert contact.normal_force is None
    assert contact.tangential_force is None


def test_penetration_depth_is_non_negative_and_positive_dist_becomes_zero() -> None:
    backend = _backend_with_fake_geom_mapping()
    penetrating = backend._convert_mujoco_contact(_fake_contact(1, 2, dist=-0.125))
    separated = backend._convert_mujoco_contact(_fake_contact(1, 2, dist=0.5))

    assert penetrating is not None
    assert separated is not None
    assert penetrating.penetration_depth == pytest.approx(0.125)
    assert separated.penetration_depth == pytest.approx(0.0)


def test_position_is_python_tuple_and_multiple_points_are_not_deduplicated() -> None:
    backend = _backend_with_fake_geom_mapping()
    backend._data.contact = (
        _fake_contact(1, 2, pos=(0.0, 0.0, 0.0)),
        _fake_contact(1, 2, pos=(1.0, 0.0, 0.0)),
    )
    backend._data.ncon = 2

    contacts = backend.get_contacts()
    assert len(contacts) == 2
    assert all(isinstance(contact.position, tuple) for contact in contacts)
    assert {contact.position for contact in contacts} == {(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)}


def test_contact_sorting_is_stable() -> None:
    backend = _backend_with_fake_geom_mapping()
    first = backend._convert_mujoco_contact(_fake_contact(1, 2, pos=(1.0, 0.0, 0.0)))
    second = backend._convert_mujoco_contact(_fake_contact(1, 2, pos=(0.0, 0.0, 0.0)))

    assert first is not None
    assert second is not None
    assert sorted((first, second), key=backend._contact_sort_key) == [second, first]


def test_non_finite_contact_data_is_rejected() -> None:
    backend = _backend_with_fake_geom_mapping()
    with pytest.raises(MuJoCoRuntimeError, match="position"):
        backend._convert_mujoco_contact(_fake_contact(1, 2, pos=(0.0, math.inf, 0.0)))


def test_visual_geom_in_contact_fails_as_unmapped_backend_error() -> None:
    geometry = BoxGeometry((1.0, 1.0, 1.0))
    body = RigidBodySpec(
        "body",
        "body",
        "dynamic",
        Transform.identity(),
        (VisualSpec("visual", geometry),),
        (ColliderSpec("collider", geometry),),
        mass_properties=create_box("tmp", (1.0, 1.0, 1.0), mass=1.0).mass_properties,
    )
    asset = create_single_body_asset(asset_id="asset", body=body)
    other_asset = create_single_body_asset(
        asset_id="other_asset",
        body=create_box("other_body", (1.0, 1.0, 1.0), mass=1.0),
    )
    scene = create_scene(
        scene_id="visual_geom_mapping",
        instances=(
            AssetInstanceSpec("body_01", asset),
            AssetInstanceSpec("other_01", other_asset, Transform(position=(0.0, 0.0, 2.0))),
        ),
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    visual_geom_ids = [
        geom_id
        for geom_id in range(int(backend._model.ngeom))
        if int(backend._model.geom_contype[geom_id]) == 0
    ]

    assert visual_geom_ids
    with pytest.raises(MuJoCoRuntimeError, match="unmapped collision geom"):
        backend._get_runtime_body_for_geom_id(visual_geom_ids[0])
