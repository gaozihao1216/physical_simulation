from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from tests.helpers.contact_wrench_math import aggregate_wrenches_for_body, force_on_body
from tests.helpers.mujoco_contact_scenes import (
    BOX_BODY_ID,
    SPHERE_BODY_ID,
    find_first_offcenter_impact,
    run_v_groove_to_rest,
)


def _v_groove_snapshot():
    backend, state, wrenches = run_v_groove_to_rest()
    try:
        net_force, net_torque = aggregate_wrenches_for_body(
            wrenches,
            runtime_body_id=SPHERE_BODY_ID,
            center_world=state.position,
        )
        return {
            "support_bodies": tuple(
                sorted(
                    wrench.contact.body_a if wrench.contact.body_b == SPHERE_BODY_ID else wrench.contact.body_b
                    for wrench in wrenches
                )
            ),
            "contact_positions": tuple(wrench.contact.position for wrench in wrenches),
            "forces": tuple(force_on_body(wrench, SPHERE_BODY_ID) for wrench in wrenches),
            "net_force": net_force,
            "net_torque": net_torque,
        }
    finally:
        backend.close()


def _impact_snapshot():
    impact = find_first_offcenter_impact(20.0)
    net_force, net_torque = aggregate_wrenches_for_body(
        impact.active_wrenches,
        runtime_body_id=BOX_BODY_ID,
        center_world=impact.impact_state.position,
    )
    return {
        "step_index": impact.step_index,
        "body_pairs": tuple((wrench.contact.body_a, wrench.contact.body_b) for wrench in impact.active_wrenches),
        "contact_positions": tuple(wrench.contact.position for wrench in impact.active_wrenches),
        "net_force": net_force,
        "net_torque": net_torque,
        "post_angular_velocity": impact.post_impact_state.angular_velocity,
    }


def test_v_groove_multidirectional_contact_is_deterministic() -> None:
    first = _v_groove_snapshot()
    second = _v_groove_snapshot()

    assert first["support_bodies"] == second["support_bodies"]
    assert len(first["forces"]) == len(second["forces"])
    for key in ("contact_positions", "forces"):
        for first_value, second_value in zip(first[key], second[key]):
            assert first_value == pytest.approx(second_value, abs=1.0e-10)
    assert first["net_force"] == pytest.approx(second["net_force"], abs=1.0e-10)
    assert first["net_torque"] == pytest.approx(second["net_torque"], abs=1.0e-10)


def test_offcenter_impact_detection_is_deterministic() -> None:
    first = _impact_snapshot()
    second = _impact_snapshot()

    assert first["step_index"] == second["step_index"]
    assert first["body_pairs"] == second["body_pairs"]
    for first_position, second_position in zip(first["contact_positions"], second["contact_positions"]):
        assert first_position == pytest.approx(second_position, abs=1.0e-10)
    assert first["net_force"] == pytest.approx(second["net_force"], abs=1.0e-10)
    assert first["net_torque"] == pytest.approx(second["net_torque"], abs=1.0e-10)
    assert first["post_angular_velocity"] == pytest.approx(second["post_angular_velocity"], abs=1.0e-10)
