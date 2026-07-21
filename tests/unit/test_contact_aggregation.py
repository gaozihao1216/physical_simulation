import pytest

from physical_simulation.runtime import (
    BodyContactWrench,
    BodyPairContactWrench,
    ContactPoint,
    ContactWrench,
    aggregate_contact_wrenches_by_body,
    aggregate_contact_wrenches_by_body_pair,
    force_on_body,
    integrate_body_contact_impulse,
    torque_about_center_from_wrench,
)
from physical_simulation.validation.errors import InvalidRuntimeStateError


def _wrench(
    *,
    body_a: str = "body_a",
    body_b: str = "body_b",
    position=(1.0, 0.0, 0.0),
    force_on_a=(0.0, 0.0, -2.0),
    pure_torque_on_a=(0.0, 0.0, 0.0),
) -> ContactWrench:
    force_on_b = tuple(-value for value in force_on_a)
    pure_torque_on_b = tuple(-value for value in pure_torque_on_a)
    return ContactWrench(
        contact=ContactPoint(body_a, body_b, position, (0.0, 0.0, 1.0), 0.0),
        force_on_body_a_world=force_on_a,
        contact_torque_on_body_a_world=pure_torque_on_a,
        force_on_body_b_world=force_on_b,
        contact_torque_on_body_b_world=pure_torque_on_b,
        normal_force_magnitude=2.0,
        tangential_force_magnitude=0.0,
    )


def test_force_and_torque_about_center_from_single_wrench() -> None:
    wrench = _wrench(position=(1.0, 0.0, 0.0), force_on_a=(0.0, 2.0, 0.0))

    assert force_on_body(wrench, "body_a") == pytest.approx((0.0, 2.0, 0.0))
    assert force_on_body(wrench, "body_b") == pytest.approx((0.0, -2.0, 0.0))
    assert torque_about_center_from_wrench(
        wrench,
        runtime_body_id="body_a",
        center_world=(0.0, 0.0, 0.0),
    ) == pytest.approx((0.0, 0.0, 2.0))


def test_aggregate_contact_wrenches_by_body() -> None:
    wrenches = (
        _wrench(position=(1.0, 0.0, 0.0), force_on_a=(0.0, 2.0, 0.0)),
        _wrench(position=(0.0, 1.0, 0.0), force_on_a=(1.0, 0.0, 0.0)),
    )

    aggregates = aggregate_contact_wrenches_by_body(
        wrenches,
        {"body_a": (0.0, 0.0, 0.0), "body_b": (0.0, 0.0, 0.0)},
    )
    by_body = {aggregate.body_id: aggregate for aggregate in aggregates}

    assert by_body["body_a"].net_force_world == pytest.approx((1.0, 2.0, 0.0))
    assert by_body["body_a"].net_torque_world == pytest.approx((0.0, 0.0, 1.0))
    assert by_body["body_a"].contact_count == 2
    assert by_body["body_b"].net_force_world == pytest.approx((-1.0, -2.0, 0.0))


def test_aggregate_contact_wrenches_by_body_pair_uses_stable_order() -> None:
    wrenches = (
        _wrench(body_a="z_body", body_b="a_body", force_on_a=(0.0, 0.0, 3.0)),
        _wrench(body_a="a_body", body_b="z_body", force_on_a=(0.0, 0.0, -1.0)),
    )

    aggregate = aggregate_contact_wrenches_by_body_pair(
        wrenches,
        {"a_body": (0.0, 0.0, 0.0), "z_body": (0.0, 0.0, 0.0)},
    )[0]

    assert aggregate.body_a == "a_body"
    assert aggregate.body_b == "z_body"
    assert aggregate.force_on_body_a_world == pytest.approx((0.0, 0.0, -4.0))
    assert aggregate.force_on_body_b_world == pytest.approx((0.0, 0.0, 4.0))
    assert aggregate.contact_count == 2


def test_integrate_body_contact_impulse() -> None:
    samples = (
        BodyContactWrench("body", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 2.0), 1),
        BodyContactWrench("body", (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 0.0, 4.0), 1),
    )

    impulse = integrate_body_contact_impulse(samples, timestep=0.5)

    assert impulse.body_id == "body"
    assert impulse.linear_impulse_world == pytest.approx((2.0, 0.0, 0.0))
    assert impulse.angular_impulse_world == pytest.approx((0.0, 0.0, 3.0))
    assert impulse.duration == pytest.approx(1.0)
    assert impulse.sample_count == 2


def test_empty_impulse_requires_body_id() -> None:
    with pytest.raises(InvalidRuntimeStateError, match="body_id"):
        integrate_body_contact_impulse((), timestep=0.1)
    assert integrate_body_contact_impulse((), timestep=0.1, body_id="body").sample_count == 0


def test_aggregate_serialization_round_trip_and_validation() -> None:
    body = BodyContactWrench("body", (0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), 2)
    pair = BodyPairContactWrench(
        "a",
        "b",
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        1,
    )

    assert BodyContactWrench.from_dict(body.to_dict()) == body
    assert BodyPairContactWrench.from_dict(pair.to_dict()) == pair
    with pytest.raises(InvalidRuntimeStateError, match="contact_count"):
        BodyContactWrench("body", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0)
    with pytest.raises(InvalidRuntimeStateError, match="center"):
        aggregate_contact_wrenches_by_body((_wrench(),), {"body_a": (0.0, 0.0, 0.0)})
