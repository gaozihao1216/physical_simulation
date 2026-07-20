from __future__ import annotations

import math

import pytest

pytest.importorskip("mujoco")

from physical_simulation.evaluation import quaternion_angular_distance
from tests.helpers.contact_wrench_math import (
    aggregate_wrenches_for_body,
    cross,
    force_on_body,
    norm,
    pure_contact_torque_on_body,
)
from tests.helpers.mujoco_contact_scenes import BOX_BODY_ID, find_first_offcenter_impact


def test_tilted_box_first_impact_produces_offcenter_com_torque_and_rotation() -> None:
    impact = find_first_offcenter_impact(20.0)
    net_force, net_torque = aggregate_wrenches_for_body(
        impact.active_wrenches,
        runtime_body_id=BOX_BODY_ID,
        center_world=impact.impact_state.position,
    )
    delta_omega = tuple(
        impact.post_impact_state.angular_velocity[index] - impact.previous_state.angular_velocity[index]
        for index in range(3)
    )

    assert impact.step_index == 97
    assert impact.time == pytest.approx(97.0 / 240.0)
    assert norm(impact.previous_state.angular_velocity) < 1.0e-6
    assert norm(impact.post_impact_state.angular_velocity) > 1.0e-3
    assert net_force[2] > 0.0
    assert norm(net_torque) > 0.05
    assert sum(net_torque[index] * delta_omega[index] for index in range(3)) > 0.0
    assert quaternion_angular_distance(
        impact.impact_state.rotation,
        impact.post_impact_state.rotation,
    ) > 1.0e-3

    pure_torque = [0.0, 0.0, 0.0]
    max_lever_arm_distance = 0.0
    max_horizontal_offset = 0.0
    for wrench in impact.active_wrenches:
        force = force_on_body(wrench, BOX_BODY_ID)
        lever = tuple(
            wrench.contact.position[index] - impact.impact_state.position[index]
            for index in range(3)
        )
        r_cross_f = cross(lever, force)
        max_lever_arm_distance = max(max_lever_arm_distance, norm(r_cross_f) / norm(force))
        max_horizontal_offset = max(max_horizontal_offset, abs(lever[0]))
        contact_torque = pure_contact_torque_on_body(wrench, BOX_BODY_ID)
        for axis in range(3):
            pure_torque[axis] += contact_torque[axis]
        assert all(math.isfinite(value) for value in (*wrench.contact.position, *force, *r_cross_f))

    assert max_horizontal_offset > 0.1
    assert max_lever_arm_distance > 0.03
    assert norm(tuple(pure_torque)) < 1.0e-8
    assert norm(net_torque) > 1000.0 * max(norm(tuple(pure_torque)), 1.0e-12)
