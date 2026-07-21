from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_ground, create_single_body_asset, create_sphere
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.runtime import integrate_body_contact_impulse, make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _settled_sphere_backend() -> tuple[MuJoCoBackend, str, str]:
    ground_asset = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    sphere_asset = create_single_body_asset(
        asset_id="sphere_asset",
        body=create_sphere("sphere_body", 0.1, mass=1.0),
    )
    scene = create_scene(
        scene_id="aggregate_support_wrench",
        instances=(
            AssetInstanceSpec("ground_01", ground_asset, fixed_base=True),
            AssetInstanceSpec("sphere_01", sphere_asset, Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=1.0 / 240.0,
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    backend.reset()
    for _ in range(720):
        backend.step()
    return backend, make_runtime_body_id("sphere_01", "sphere_body"), make_runtime_body_id("ground_01", "ground_body")


def test_backend_body_contact_wrench_aggregation_supports_weight() -> None:
    backend, sphere_id, ground_id = _settled_sphere_backend()

    aggregates = {aggregate.body_id: aggregate for aggregate in backend.get_body_contact_wrenches()}

    assert set(aggregates) == {ground_id, sphere_id}
    assert aggregates[sphere_id].contact_count >= 1
    assert aggregates[sphere_id].net_force_world[2] == pytest.approx(9.81, rel=0.05, abs=0.1)
    assert aggregates[ground_id].net_force_world[2] == pytest.approx(-9.81, rel=0.05, abs=0.1)
    assert aggregates[sphere_id].net_torque_world == pytest.approx((0.0, 0.0, 0.0), abs=0.1)
    backend.close()


def test_backend_body_pair_contact_wrench_aggregation_is_equal_and_opposite() -> None:
    backend, sphere_id, ground_id = _settled_sphere_backend()

    pair = backend.get_body_pair_contact_wrenches()[0]

    assert {pair.body_a, pair.body_b} == {ground_id, sphere_id}
    assert pair.contact_count >= 1
    assert tuple(pair.force_on_body_a_world[index] + pair.force_on_body_b_world[index] for index in range(3)) == pytest.approx(
        (0.0, 0.0, 0.0),
        abs=1.0e-8,
    )
    backend.close()


def test_discrete_impulse_integrates_backend_body_contact_wrench_samples() -> None:
    backend, sphere_id, _ground_id = _settled_sphere_backend()
    timestep = backend.scene.timestep
    samples = []
    for _ in range(10):
        samples.extend(aggregate for aggregate in backend.get_body_contact_wrenches() if aggregate.body_id == sphere_id)
        backend.step()

    impulse = integrate_body_contact_impulse(samples, timestep=timestep, body_id=sphere_id)

    assert impulse.sample_count == 10
    assert impulse.duration == pytest.approx(10 * timestep)
    assert impulse.linear_impulse_world[2] == pytest.approx(9.81 * 10 * timestep, rel=0.08, abs=0.05)
    backend.close()
