from __future__ import annotations

import math

import pytest

from physical_simulation.evaluation import (
    BodyStateSample,
    SettlingCriteria,
    evaluate_resting_contact,
    quaternion_angular_distance,
)
from physical_simulation.runtime import ContactPoint, RigidBodyState


def _state(
    *,
    z: float = 0.2,
    position=(0.0, 0.0, 0.2),
    rotation=(1.0, 0.0, 0.0, 0.0),
    linear=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
) -> RigidBodyState:
    if position == (0.0, 0.0, 0.2):
        position = (0.0, 0.0, z)
    return RigidBodyState(
        body_id="body",
        position=position,
        rotation=rotation,
        linear_velocity=linear,
        angular_velocity=angular,
    )


def _contact(depth: float = 0.01, other: str = "ground") -> ContactPoint:
    return ContactPoint(
        body_a="body",
        body_b=other,
        position=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        penetration_depth=depth,
    )


def _sample(index: int, state: RigidBodyState, contacts=()) -> BodyStateSample:
    return BodyStateSample(float(index), index, state, tuple(contacts))


def test_settling_criteria_validation() -> None:
    assert SettlingCriteria().window_steps == 120
    with pytest.raises(ValueError, match="window_steps"):
        SettlingCriteria(window_steps=1)
    with pytest.raises(ValueError, match="max_linear_speed"):
        SettlingCriteria(max_linear_speed=-1.0)
    with pytest.raises(ValueError, match="max_angular_speed"):
        SettlingCriteria(max_angular_speed=math.inf)


def test_stationary_contact_is_settled_and_metrics_are_computed() -> None:
    samples = (
        _sample(0, _state(z=1.0), ()),
        _sample(1, _state(z=0.2), (_contact(0.01), _contact(0.03))),
        _sample(2, _state(z=0.2), (_contact(0.02),)),
    )
    metrics = evaluate_resting_contact(samples, "body", criteria=SettlingCriteria(window_steps=2))

    assert metrics.initial_height == pytest.approx(1.0)
    assert metrics.final_height == pytest.approx(0.2)
    assert metrics.minimum_height == pytest.approx(0.2)
    assert metrics.maximum_penetration_depth == pytest.approx(0.03)
    assert metrics.contact_step_count == 2
    assert metrics.final_contact_count == 1
    assert metrics.settled


def test_high_speed_position_drift_and_angular_speed_are_not_settled() -> None:
    criteria = SettlingCriteria(window_steps=2, max_linear_speed=0.02, max_angular_speed=0.05, max_position_drift=0.002)
    high_speed = (
        _sample(0, _state(linear=(0.0, 0.0, 0.0)), (_contact(),)),
        _sample(1, _state(linear=(0.1, 0.0, 0.0)), (_contact(),)),
    )
    drifting = (
        _sample(0, _state(position=(0.0, 0.0, 0.2)), (_contact(),)),
        _sample(1, _state(position=(0.01, 0.0, 0.2)), (_contact(),)),
    )
    spinning = (
        _sample(0, _state(angular=(0.0, 0.0, 0.0)), (_contact(),)),
        _sample(1, _state(angular=(0.0, 0.0, 0.2)), (_contact(),)),
    )

    assert not evaluate_resting_contact(high_speed, "body", criteria=criteria).settled
    assert not evaluate_resting_contact(drifting, "body", criteria=criteria).settled
    assert not evaluate_resting_contact(spinning, "body", criteria=criteria).settled


def test_final_contact_requirement_is_configurable() -> None:
    samples = (
        _sample(0, _state(), (_contact(),)),
        _sample(1, _state(), ()),
    )
    assert not evaluate_resting_contact(samples, "body", criteria=SettlingCriteria(window_steps=2)).settled
    assert evaluate_resting_contact(
        samples,
        "body",
        criteria=SettlingCriteria(window_steps=2, require_final_contact=False),
    ).settled


def test_quaternion_sign_equivalence_has_zero_orientation_drift() -> None:
    assert quaternion_angular_distance((1.0, 0.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0)
    samples = (
        _sample(0, _state(rotation=(1.0, 0.0, 0.0, 0.0)), (_contact(),)),
        _sample(1, _state(rotation=(-1.0, 0.0, 0.0, 0.0)), (_contact(),)),
    )
    metrics = evaluate_resting_contact(samples, "body", criteria=SettlingCriteria(window_steps=2))
    assert metrics.orientation_drift_last_window == pytest.approx(0.0)


def test_invalid_samples_raise() -> None:
    with pytest.raises(ValueError, match="at least two"):
        evaluate_resting_contact((), "body")
    with pytest.raises(ValueError, match="does not match"):
        evaluate_resting_contact(
            (_sample(0, _state()), _sample(1, RigidBodyState("other", (0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0)))),
            "body",
        )
