from __future__ import annotations

import math

import pytest

from physical_simulation.runtime import ContactPoint, ContactWrench
from physical_simulation.validation.errors import InvalidRuntimeStateError


def _contact() -> ContactPoint:
    return ContactPoint(
        body_a="body_a",
        body_b="body_b",
        position=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        penetration_depth=0.0,
    )


def _wrench(**overrides) -> ContactWrench:
    fields = {
        "contact": _contact(),
        "force_on_body_a_world": (0.0, 0.0, -1.0),
        "contact_torque_on_body_a_world": (0.0, 0.0, 0.0),
        "force_on_body_b_world": (0.0, 0.0, 1.0),
        "contact_torque_on_body_b_world": (0.0, 0.0, 0.0),
        "normal_force_magnitude": 1.0,
        "tangential_force_magnitude": 0.0,
    }
    fields.update(overrides)
    return ContactWrench(**fields)


def test_contact_wrench_accepts_valid_values_and_normalizes_public_tuples() -> None:
    wrench = _wrench(force_on_body_a_world=[0, 0, -1])

    assert wrench.force_on_body_a_world == (0.0, 0.0, -1.0)
    assert isinstance(wrench.force_on_body_a_world, tuple)
    assert wrench.to_dict()["force_on_body_b_world"] == [0.0, 0.0, 1.0]
    assert ContactWrench.from_dict(wrench.to_dict()) == wrench


def test_contact_wrench_is_frozen() -> None:
    wrench = _wrench()

    with pytest.raises(Exception):
        wrench.normal_force_magnitude = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("force_on_body_a_world", (math.inf, 0.0, 0.0)),
        ("contact_torque_on_body_a_world", (0.0, math.nan, 0.0)),
        ("force_on_body_b_world", (0.0, 0.0)),
        ("contact_torque_on_body_b_world", (0.0, 0.0, math.inf)),
    ),
)
def test_contact_wrench_rejects_invalid_vectors(field_name: str, value: object) -> None:
    with pytest.raises(InvalidRuntimeStateError, match=field_name):
        _wrench(**{field_name: value})


@pytest.mark.parametrize("field_name", ("normal_force_magnitude", "tangential_force_magnitude"))
def test_contact_wrench_rejects_negative_magnitudes(field_name: str) -> None:
    with pytest.raises(InvalidRuntimeStateError, match=field_name):
        _wrench(**{field_name: -1.0})


def test_contact_wrench_requires_contact_point() -> None:
    with pytest.raises(InvalidRuntimeStateError, match="contact must be ContactPoint"):
        _wrench(contact=object())
