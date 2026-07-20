from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from physical_simulation.assets import Transform, create_box, create_single_body_asset
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.scene import AssetInstanceSpec, create_scene


def test_static_body_state_stays_fixed_across_steps() -> None:
    body = create_box(
        "static_body",
        (1.0, 1.0, 1.0),
        body_type="static",
        transform=Transform(position=(0.0, 0.0, 0.5)),
    )
    asset = create_single_body_asset(asset_id="static_asset", body=body)
    scene = create_scene(scene_id="static_body", instances=(AssetInstanceSpec("static_01", asset),))
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    initial = backend.reset().get_body_state("static_01/static_body")

    for _ in range(10):
        result = backend.step()
    state = result.get_body_state("static_01/static_body")

    assert state.position == pytest.approx(initial.position)
    assert state.rotation == pytest.approx(initial.rotation)
    assert state.linear_velocity == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-12)
    assert state.angular_velocity == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-12)
