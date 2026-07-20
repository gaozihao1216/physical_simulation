from __future__ import annotations

import math

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_box, create_ground, create_single_body_asset
from physical_simulation.backends import BackendNotLoadedError, MuJoCoBackend
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _box_drop_scene(scene_id: str = "contact_wrench_box"):
    ground_asset = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box_asset = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.4, 0.4, 0.4), mass=1.0),
    )
    return create_scene(
        scene_id=scene_id,
        instances=(
            AssetInstanceSpec("ground_01", ground_asset, fixed_base=True),
            AssetInstanceSpec("box_01", box_asset, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=1.0 / 240.0,
    )


def _force_on_body(wrench, runtime_body_id: str) -> tuple[float, float, float] | None:
    if wrench.contact.body_a == runtime_body_id:
        return wrench.force_on_body_a_world
    if wrench.contact.body_b == runtime_body_id:
        return wrench.force_on_body_b_world
    return None


def _sum_force_on_body(wrenches, runtime_body_id: str) -> tuple[float, float, float]:
    total = [0.0, 0.0, 0.0]
    for wrench in wrenches:
        force = _force_on_body(wrench, runtime_body_id)
        if force is None:
            continue
        for axis in range(3):
            total[axis] += force[axis]
    return tuple(total)


def _assert_equal_and_opposite_wrench_sides(wrench) -> None:
    for axis in range(3):
        assert wrench.force_on_body_a_world[axis] == pytest.approx(-wrench.force_on_body_b_world[axis], abs=1.0e-9)
        assert wrench.contact_torque_on_body_a_world[axis] == pytest.approx(
            -wrench.contact_torque_on_body_b_world[axis],
            abs=1.0e-9,
        )


def test_get_contact_wrenches_lifecycle_and_no_contact_scene() -> None:
    backend = MuJoCoBackend()
    with pytest.raises(BackendNotLoadedError):
        backend.get_contact_wrenches()

    backend.load_scene(_box_drop_scene("no_contact_wrench"))
    backend.reset()
    assert backend.get_contacts() == ()
    assert backend.get_contact_wrenches() == ()

    backend.close()
    with pytest.raises(BackendNotLoadedError):
        backend.get_contact_wrenches()


def test_box_drop_contact_wrenches_match_contacts_and_support_weight() -> None:
    scene = _box_drop_scene()
    box_id = make_runtime_body_id("box_01", "box_body")
    ground_id = make_runtime_body_id("ground_01", "ground_body")
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    backend.reset()
    for _ in range(720):
        backend.step()

    contacts = backend.get_contacts()
    wrenches = backend.get_contact_wrenches()

    assert wrenches
    assert tuple(wrench.contact for wrench in wrenches) == contacts
    assert all(wrench.contact.normal_force is None for wrench in wrenches)
    assert all(wrench.contact.tangential_force is None for wrench in wrenches)
    for wrench in wrenches:
        _assert_equal_and_opposite_wrench_sides(wrench)
        assert math.isfinite(wrench.normal_force_magnitude)
        assert math.isfinite(wrench.tangential_force_magnitude)
        assert wrench.normal_force_magnitude >= 0.0
        assert wrench.tangential_force_magnitude >= 0.0
        assert wrench.contact_torque_on_body_a_world == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-8)
        assert wrench.contact_torque_on_body_b_world == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-8)

    box_force = _sum_force_on_body(wrenches, box_id)
    ground_force = _sum_force_on_body(wrenches, ground_id)
    normal_force_sum = sum(wrench.normal_force_magnitude for wrench in wrenches)
    tangential_force_sum = sum(wrench.tangential_force_magnitude for wrench in wrenches)

    assert box_force[0] == pytest.approx(0.0, abs=0.1)
    assert box_force[1] == pytest.approx(0.0, abs=0.1)
    assert box_force[2] == pytest.approx(9.81, rel=0.05, abs=0.1)
    assert ground_force == pytest.approx(tuple(-value for value in box_force), abs=1.0e-9)
    assert normal_force_sum == pytest.approx(9.81, rel=0.05, abs=0.1)
    assert tangential_force_sum == pytest.approx(0.0, abs=0.1)
    backend.close()


def test_impact_contact_wrench_has_positive_normal_force() -> None:
    scene = _box_drop_scene("impact_contact_wrench")
    box_id = make_runtime_body_id("box_01", "box_body")
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    backend.reset()

    impact = None
    for _ in range(360):
        backend.step()
        candidates = [wrench for wrench in backend.get_contact_wrenches() if wrench.normal_force_magnitude > 0.0]
        if candidates:
            impact = candidates[0]
            break

    assert impact is not None
    force_on_box = _force_on_body(impact, box_id)
    assert force_on_box is not None
    assert impact.normal_force_magnitude > 0.0
    assert all(math.isfinite(value) for value in (*force_on_box, impact.normal_force_magnitude))
    assert force_on_box[2] > 0.0
    backend.close()


def test_fixed_fixed_contact_wrench_returns_finite_zero_or_solver_wrench() -> None:
    first_asset = create_single_body_asset(
        asset_id="first_asset",
        body=create_box("first_body", (0.2, 0.2, 0.2), body_type="static"),
    )
    second_asset = create_single_body_asset(
        asset_id="second_asset",
        body=create_box("second_body", (0.2, 0.2, 0.2), body_type="static"),
    )
    scene = create_scene(
        scene_id="fixed_fixed_contact_wrench",
        instances=(
            AssetInstanceSpec("first", first_asset),
            AssetInstanceSpec("second", second_asset, Transform(position=(0.0, 0.0, 0.15))),
        ),
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    backend.reset()

    contacts = backend.get_contacts()
    wrenches = backend.get_contact_wrenches()

    assert contacts
    assert tuple(wrench.contact for wrench in wrenches) == contacts
    for wrench in wrenches:
        _assert_equal_and_opposite_wrench_sides(wrench)
        assert all(
            math.isfinite(value)
            for value in (
                *wrench.force_on_body_a_world,
                *wrench.contact_torque_on_body_a_world,
                *wrench.force_on_body_b_world,
                *wrench.contact_torque_on_body_b_world,
                wrench.normal_force_magnitude,
                wrench.tangential_force_magnitude,
            )
        )
    backend.close()
