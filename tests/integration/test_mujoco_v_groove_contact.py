from __future__ import annotations

import math

import pytest

pytest.importorskip("mujoco")

from tests.helpers.contact_wrench_math import aggregate_wrenches_for_body, force_on_body, norm
from tests.helpers.mujoco_contact_scenes import (
    LEFT_RAMP_BODY_ID,
    RIGHT_RAMP_BODY_ID,
    SPHERE_BODY_ID,
    create_v_groove_scene,
    run_v_groove_to_rest,
)


def test_v_groove_collision_masks_allow_only_sphere_ramp_contacts() -> None:
    backend, _state, wrenches = run_v_groove_to_rest()
    try:
        contacts = backend.get_contacts()
        assert contacts
        assert all(SPHERE_BODY_ID in {contact.body_a, contact.body_b} for contact in contacts)
        assert all({contact.body_a, contact.body_b} != {LEFT_RAMP_BODY_ID, RIGHT_RAMP_BODY_ID} for contact in contacts)
        assert {
            contact.body_a if contact.body_b == SPHERE_BODY_ID else contact.body_b
            for contact in contacts
        } == {LEFT_RAMP_BODY_ID, RIGHT_RAMP_BODY_ID}
        assert {
            wrench.contact.body_a if wrench.contact.body_b == SPHERE_BODY_ID else wrench.contact.body_b
            for wrench in wrenches
        } == {LEFT_RAMP_BODY_ID, RIGHT_RAMP_BODY_ID}
    finally:
        backend.close()


def test_v_groove_sphere_settles_with_near_zero_net_torque() -> None:
    backend, state, wrenches = run_v_groove_to_rest()
    try:
        assert norm(state.linear_velocity) < 1.0e-6
        assert norm(state.angular_velocity) < 1.0e-6
        net_force, net_torque = aggregate_wrenches_for_body(
            wrenches,
            runtime_body_id=SPHERE_BODY_ID,
            center_world=state.position,
        )

        assert net_force[0] == pytest.approx(0.0, abs=0.1)
        assert net_force[1] == pytest.approx(0.0, abs=0.1)
        assert net_force[2] == pytest.approx(9.81, rel=0.05, abs=0.1)
        assert norm(net_torque) < 0.02
        assert all(math.isfinite(value) for value in (*state.position, *net_force, *net_torque))
    finally:
        backend.close()


def test_v_groove_scene_compiles_deterministically() -> None:
    scene = create_v_groove_scene()

    assert scene == create_v_groove_scene()
