from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mujoco")

from physical_simulation.backends import MuJoCoBackend, MuJoCoRuntimeError
from physical_simulation.backends.mujoco_backend import _contact_frame_vector_to_world
from physical_simulation.runtime import ContactPoint


def _contact(body_a: str = "a", body_b: str = "b", normal=(0.0, 0.0, 1.0)) -> ContactPoint:
    return ContactPoint(
        body_a=body_a,
        body_b=body_b,
        position=(0.0, 0.0, 0.0),
        normal=normal,
        penetration_depth=0.0,
    )


def test_contact_frame_identity_transform() -> None:
    assert _contact_frame_vector_to_world(
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        (1.0, 2.0, 3.0),
    ) == pytest.approx((1.0, 2.0, 3.0))


def test_contact_frame_uses_rows_transpose_for_world_transform() -> None:
    frame = (
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )

    assert _contact_frame_vector_to_world(frame, (2.0, 3.0, 4.0)) == pytest.approx((3.0, 4.0, 2.0))


@pytest.mark.parametrize(
    ("frame", "vector"),
    (
        ((1.0, 0.0), (1.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (1.0, 0.0)),
        ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (1.0, float("nan"), 0.0)),
    ),
)
def test_contact_frame_transform_rejects_invalid_inputs(frame: object, vector: object) -> None:
    with pytest.raises(MuJoCoRuntimeError):
        _contact_frame_vector_to_world(frame, vector)


def test_wrench_assignment_handles_unswapped_public_order() -> None:
    backend = MuJoCoBackend()
    force_a, torque_a, force_b, torque_b = backend._assign_wrench_to_public_bodies(
        contact_point=_contact("a", "b"),
        geom1_body_id="a",
        geom2_body_id="b",
        force_on_geom2_world=(0.0, 0.0, 5.0),
        torque_on_geom2_world=(1.0, 2.0, 3.0),
    )

    assert force_a == pytest.approx((0.0, 0.0, -5.0))
    assert force_b == pytest.approx((0.0, 0.0, 5.0))
    assert torque_a == pytest.approx((-1.0, -2.0, -3.0))
    assert torque_b == pytest.approx((1.0, 2.0, 3.0))


def test_wrench_assignment_handles_swapped_public_order() -> None:
    backend = MuJoCoBackend()
    force_a, torque_a, force_b, torque_b = backend._assign_wrench_to_public_bodies(
        contact_point=_contact("a", "b"),
        geom1_body_id="b",
        geom2_body_id="a",
        force_on_geom2_world=(0.0, 0.0, -5.0),
        torque_on_geom2_world=(-1.0, -2.0, -3.0),
    )

    assert force_a == pytest.approx((0.0, 0.0, -5.0))
    assert force_b == pytest.approx((0.0, 0.0, 5.0))
    assert torque_a == pytest.approx((-1.0, -2.0, -3.0))
    assert torque_b == pytest.approx((1.0, 2.0, 3.0))


def test_force_decomposition_uses_public_normal_and_world_force() -> None:
    backend = MuJoCoBackend()
    mapped = SimpleNamespace(contact_point=_contact(normal=(0.0, 0.0, 1.0)), contact_index=7, geom1_id=1, geom2_id=2)

    normal, tangent = backend._decompose_contact_force(mapped, (3.0, 4.0, 10.0))

    assert normal == pytest.approx(10.0)
    assert tangent == pytest.approx(5.0)


def test_force_decomposition_rejects_clearly_negative_normal_force() -> None:
    backend = MuJoCoBackend()
    backend._scene = SimpleNamespace(scene_id="negative")
    backend._data = SimpleNamespace(time=0.0)
    mapped = SimpleNamespace(
        contact_point=_contact(normal=(0.0, 0.0, 1.0)),
        contact_index=7,
        geom1_id=1,
        geom2_id=2,
    )

    with pytest.raises(MuJoCoRuntimeError, match="unexpected negative direction"):
        backend._decompose_contact_force(mapped, (0.0, 0.0, -1.0))
