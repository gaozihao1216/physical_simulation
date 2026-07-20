from __future__ import annotations

import pytest

from physical_simulation.runtime import ContactPoint, ContactWrench
from tests.helpers.contact_wrench_math import (
    aggregate_wrenches_for_body,
    force_on_body,
    norm,
    torque_about_center_from_wrench,
)


def _wrench(
    *,
    position=(0.0, 0.0, 0.0),
    force_on_target=(0.0, 0.0, 1.0),
    pure_torque_on_target=(0.0, 0.0, 0.0),
    target: str = "body",
    other: str = "support",
) -> ContactWrench:
    return ContactWrench(
        contact=ContactPoint(
            body_a=target,
            body_b=other,
            position=position,
            normal=(0.0, 0.0, -1.0),
            penetration_depth=0.0,
        ),
        force_on_body_a_world=force_on_target,
        contact_torque_on_body_a_world=pure_torque_on_target,
        force_on_body_b_world=tuple(-value for value in force_on_target),
        contact_torque_on_body_b_world=tuple(-value for value in pure_torque_on_target),
        normal_force_magnitude=max(0.0, -force_on_target[2]),
        tangential_force_magnitude=0.0,
    )


def test_aggregate_net_force_sums_multiple_contacts() -> None:
    first = _wrench(force_on_target=(1.0, 0.0, 2.0))
    second = _wrench(force_on_target=(-1.0, 0.0, 3.0))

    net_force, net_torque = aggregate_wrenches_for_body(
        (first, second),
        runtime_body_id="body",
        center_world=(0.0, 0.0, 0.0),
    )

    assert net_force == pytest.approx((0.0, 0.0, 5.0))
    assert net_torque == pytest.approx((0.0, 0.0, 0.0))


def test_torque_about_center_uses_right_hand_cross_product() -> None:
    wrench = _wrench(position=(1.0, 0.0, 0.0), force_on_target=(0.0, 0.0, 10.0))

    torque = torque_about_center_from_wrench(
        wrench,
        runtime_body_id="body",
        center_world=(0.0, 0.0, 0.0),
    )

    assert torque == pytest.approx((0.0, -10.0, 0.0))


def test_symmetric_contacts_cancel_y_axis_torque() -> None:
    first = _wrench(position=(1.0, 0.0, -0.2), force_on_target=(0.0, 0.0, 5.0))
    second = _wrench(position=(-1.0, 0.0, -0.2), force_on_target=(0.0, 0.0, 5.0))

    net_force, net_torque = aggregate_wrenches_for_body(
        (first, second),
        runtime_body_id="body",
        center_world=(0.0, 0.0, 0.0),
    )

    assert net_force == pytest.approx((0.0, 0.0, 10.0))
    assert net_torque[1] == pytest.approx(0.0)


def test_off_center_single_contact_has_nonzero_torque() -> None:
    wrench = _wrench(position=(0.25, 0.0, -0.2), force_on_target=(0.0, 0.0, 10.0))

    torque = torque_about_center_from_wrench(
        wrench,
        runtime_body_id="body",
        center_world=(0.0, 0.0, 0.0),
    )

    assert norm(torque) > 0.0
    assert torque[1] < 0.0


def test_force_on_body_rejects_unrelated_body() -> None:
    with pytest.raises(ValueError, match="does not involve"):
        force_on_body(_wrench(), "other_body")
