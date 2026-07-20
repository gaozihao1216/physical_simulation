from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from tests.helpers.contact_wrench_math import aggregate_wrenches_for_body, norm
from tests.helpers.mujoco_contact_scenes import BOX_BODY_ID, find_first_offcenter_impact


def test_mirrored_offcenter_impacts_flip_torque_and_rotation_direction() -> None:
    positive = find_first_offcenter_impact(20.0)
    negative = find_first_offcenter_impact(-20.0)
    _positive_force, positive_torque = aggregate_wrenches_for_body(
        positive.active_wrenches,
        runtime_body_id=BOX_BODY_ID,
        center_world=positive.impact_state.position,
    )
    _negative_force, negative_torque = aggregate_wrenches_for_body(
        negative.active_wrenches,
        runtime_body_id=BOX_BODY_ID,
        center_world=negative.impact_state.position,
    )

    assert positive.step_index == negative.step_index
    assert positive_torque[1] < 0.0
    assert negative_torque[1] > 0.0
    assert positive.post_impact_state.angular_velocity[1] < 0.0
    assert negative.post_impact_state.angular_velocity[1] > 0.0
    assert abs(positive_torque[1]) == pytest.approx(abs(negative_torque[1]), rel=0.2)
    assert abs(positive.post_impact_state.angular_velocity[1]) == pytest.approx(
        abs(negative.post_impact_state.angular_velocity[1]),
        rel=0.2,
    )
    assert abs(positive_torque[1]) > 5.0 * max(abs(positive_torque[0]), abs(positive_torque[2]), 1.0e-12)
    assert abs(negative_torque[1]) > 5.0 * max(abs(negative_torque[0]), abs(negative_torque[2]), 1.0e-12)

    positive_offsets = sorted(
        wrench.contact.position[0] - positive.impact_state.position[0]
        for wrench in positive.active_wrenches
    )
    negative_offsets = sorted(
        wrench.contact.position[0] - negative.impact_state.position[0]
        for wrench in negative.active_wrenches
    )

    assert len(positive_offsets) == len(negative_offsets)
    for positive_offset, negative_offset in zip(positive_offsets, reversed(negative_offsets)):
        assert positive_offset == pytest.approx(-negative_offset, rel=0.2, abs=1.0e-6)
    assert norm(positive_torque) == pytest.approx(norm(negative_torque), rel=0.2)
