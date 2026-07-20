from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from tests.helpers.contact_wrench_math import aggregate_wrenches_for_body, dot, force_on_body, norm, normalized
from tests.helpers.mujoco_contact_scenes import (
    LEFT_RAMP_BODY_ID,
    RIGHT_RAMP_BODY_ID,
    SPHERE_BODY_ID,
    run_v_groove_to_rest,
)


def _net_force_from_external_body(wrenches, external_body_id: str):
    total = [0.0, 0.0, 0.0]
    for wrench in wrenches:
        if external_body_id not in (wrench.contact.body_a, wrench.contact.body_b):
            continue
        force = force_on_body(wrench, SPHERE_BODY_ID)
        for axis in range(3):
            total[axis] += force[axis]
    return tuple(total)


def test_v_groove_contact_forces_are_multidirectional_and_nonparallel() -> None:
    backend, state, wrenches = run_v_groove_to_rest()
    try:
        left_force = _net_force_from_external_body(wrenches, LEFT_RAMP_BODY_ID)
        right_force = _net_force_from_external_body(wrenches, RIGHT_RAMP_BODY_ID)
        left_direction = normalized(left_force)
        right_direction = normalized(right_force)

        assert left_force[2] > 0.0
        assert right_force[2] > 0.0
        assert left_force[0] > 0.1
        assert right_force[0] < -0.1
        assert abs(dot(left_direction, right_direction)) < 0.95

        net_force, _net_torque = aggregate_wrenches_for_body(
            wrenches,
            runtime_body_id=SPHERE_BODY_ID,
            center_world=state.position,
        )
        assert net_force == pytest.approx((0.0, 0.0, 9.81), abs=0.1)
    finally:
        backend.close()


def test_v_groove_has_two_distinct_supporting_runtime_bodies_not_just_two_points() -> None:
    backend, _state, wrenches = run_v_groove_to_rest()
    try:
        external_body_ids = {
            wrench.contact.body_a if wrench.contact.body_b == SPHERE_BODY_ID else wrench.contact.body_b
            for wrench in wrenches
        }

        assert external_body_ids == {LEFT_RAMP_BODY_ID, RIGHT_RAMP_BODY_ID}
        assert len(external_body_ids) >= 2
        assert all(norm(force_on_body(wrench, SPHERE_BODY_ID)) > 0.0 for wrench in wrenches)
    finally:
        backend.close()
