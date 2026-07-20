from __future__ import annotations

import math

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import (
    BoxGeometry,
    ColliderSpec,
    RigidBodySpec,
    Transform,
    create_ground,
    create_single_body_asset,
    create_sphere,
)
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


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


def test_sphere_resting_contact_wrench_supports_weight() -> None:
    ground_asset = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    sphere_asset = create_single_body_asset(
        asset_id="sphere_asset",
        body=create_sphere("sphere_body", 0.1, mass=1.0),
    )
    scene = create_scene(
        scene_id="sphere_support_wrench",
        instances=(
            AssetInstanceSpec("ground_01", ground_asset, fixed_base=True),
            AssetInstanceSpec("sphere_01", sphere_asset, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=1.0 / 240.0,
    )
    sphere_id = make_runtime_body_id("sphere_01", "sphere_body")
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    backend.reset()
    for _ in range(720):
        backend.step()

    wrenches = backend.get_contact_wrenches()
    sphere_force = _sum_force_on_body(wrenches, sphere_id)
    normal_force_sum = sum(wrench.normal_force_magnitude for wrench in wrenches if sphere_id in {wrench.contact.body_a, wrench.contact.body_b})
    tangential_force_sum = sum(wrench.tangential_force_magnitude for wrench in wrenches if sphere_id in {wrench.contact.body_a, wrench.contact.body_b})

    assert wrenches
    assert all(math.isfinite(value) for wrench in wrenches for value in (*wrench.force_on_body_a_world, *wrench.force_on_body_b_world))
    assert sphere_force[0] == pytest.approx(0.0, abs=0.1)
    assert sphere_force[1] == pytest.approx(0.0, abs=0.1)
    assert sphere_force[2] == pytest.approx(9.81, rel=0.05, abs=0.1)
    assert normal_force_sum == pytest.approx(9.81, rel=0.05, abs=0.1)
    assert tangential_force_sum == pytest.approx(0.0, abs=0.1)
    backend.close()


def test_compound_surface_wrench_maps_all_table_colliders_to_one_runtime_body() -> None:
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
        scene_id="compound_surface_support_wrench",
        instances=(
            AssetInstanceSpec("table_01", table_asset, fixed_base=True),
            AssetInstanceSpec("sphere_01", sphere_asset, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=1.0 / 240.0,
    )
    table_id = make_runtime_body_id("table_01", "table_body")
    sphere_id = make_runtime_body_id("sphere_01", "sphere_body")
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    backend.reset()
    for _ in range(720):
        backend.step()

    wrenches = backend.get_contact_wrenches()
    sphere_wrenches = [wrench for wrench in wrenches if sphere_id in {wrench.contact.body_a, wrench.contact.body_b}]
    sphere_force = _sum_force_on_body(sphere_wrenches, sphere_id)

    assert sphere_wrenches
    assert all(table_id in {wrench.contact.body_a, wrench.contact.body_b} for wrench in sphere_wrenches)
    assert all(wrench.contact.body_a != wrench.contact.body_b for wrench in sphere_wrenches)
    assert sphere_force[2] == pytest.approx(9.81, rel=0.05, abs=0.1)
    backend.close()
