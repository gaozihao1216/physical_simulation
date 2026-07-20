import math

import pytest

from physical_simulation.runtime import (
    ContactPoint,
    JointState,
    RigidBodyState,
    SimulationStepResult,
    make_runtime_body_id,
)
from physical_simulation.validation.errors import InvalidRuntimeStateError


def test_rigid_body_state_validation_and_round_trip() -> None:
    state = RigidBodyState("i/body", (0.0, 0.0, 1.0), (2.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert state.rotation == (1.0, 0.0, 0.0, 0.0)
    assert RigidBodyState.from_dict(state.to_dict()) == state
    with pytest.raises(InvalidRuntimeStateError, match="rotation"):
        RigidBodyState("i/body", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0), (0, 0, 0), (0, 0, 0))
    with pytest.raises(InvalidRuntimeStateError, match="position"):
        RigidBodyState("i/body", (math.nan, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0), (0, 0, 0), (0, 0, 0))


def test_joint_state_validation_and_round_trip() -> None:
    state = JointState("i/joint", (1.0, 2.0), (0.1, 0.2), (3.0, 4.0))
    assert JointState.from_dict(state.to_dict()) == state
    with pytest.raises(InvalidRuntimeStateError, match="lengths"):
        JointState("j", (1.0,), (1.0, 2.0))
    with pytest.raises(InvalidRuntimeStateError, match="applied_force"):
        JointState("j", (1.0,), (1.0,), (1.0, 2.0))
    with pytest.raises(InvalidRuntimeStateError, match="at least"):
        JointState("j", (), ())
    with pytest.raises(InvalidRuntimeStateError, match="position"):
        JointState("j", (math.inf,), (0.0,))


def test_contact_point_validation_and_round_trip() -> None:
    contact = ContactPoint("a", "b", (0.0, 0.0, 0.0), (0.0, 0.0, -2.0), 0.001)
    assert contact.normal == (0.0, 0.0, -1.0)
    assert ContactPoint.from_dict(contact.to_dict()) == contact
    with pytest.raises(InvalidRuntimeStateError, match="normal"):
        ContactPoint("a", "b", (0, 0, 0), (0, 0, 0), 0.0)
    with pytest.raises(InvalidRuntimeStateError, match="different"):
        ContactPoint("a", "a", (0, 0, 0), (0, 0, 1), 0.0)
    with pytest.raises(InvalidRuntimeStateError, match="penetration_depth"):
        ContactPoint("a", "b", (0, 0, 0), (0, 0, 1), -0.1)
    with pytest.raises(InvalidRuntimeStateError, match="normal_force"):
        ContactPoint("a", "b", (0, 0, 0), (0, 0, 1), 0.0, normal_force=-1.0)


def test_simulation_step_result_and_runtime_ids() -> None:
    body_id = make_runtime_body_id("crate_01", "body")
    state = RigidBodyState(body_id, (0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0))
    joint = JointState("crate_01/joint", (0.0,), (0.0,))
    contact = ContactPoint(body_id, "ground/body", (0, 0, 0), (0, 0, 1), 0.0)
    result = SimulationStepResult(0.1, 24, (state,), (joint,), (contact,))
    assert result.get_body_state(body_id) == state
    assert result.get_joint_state("crate_01/joint") == joint
    assert result.has_contacts
    assert SimulationStepResult.from_dict(result.to_dict()) == result
    with pytest.raises(KeyError, match="missing"):
        result.get_body_state("missing")
    with pytest.raises(InvalidRuntimeStateError, match="time"):
        SimulationStepResult(-0.1, 0, (state,))
    with pytest.raises(InvalidRuntimeStateError, match="step_index"):
        SimulationStepResult(0.0, -1, (state,))
    with pytest.raises(InvalidRuntimeStateError, match="body_id"):
        SimulationStepResult(0.0, 0, (state, state))
    with pytest.raises(InvalidRuntimeStateError, match="joint_id"):
        SimulationStepResult(0.0, 0, (state,), (joint, joint))
    with pytest.raises(InvalidRuntimeStateError, match="instance_id"):
        make_runtime_body_id("", "body")
    with pytest.raises(InvalidRuntimeStateError, match="separator"):
        make_runtime_body_id("bad/id", "body")
    assert make_runtime_body_id("crate_02", "body") != body_id
