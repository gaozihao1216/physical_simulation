"""Contact wrench aggregation and discrete impulse utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from physical_simulation.runtime.contact_wrench import ContactWrench
from physical_simulation.validation.asset_validator import _as_float_tuple, _finite_float, _non_empty_string
from physical_simulation.validation.errors import InvalidRuntimeStateError

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class BodyContactWrench:
    """Net contact wrench applied to one runtime body about a chosen world center."""

    body_id: str
    center_world: Vector3
    net_force_world: Vector3
    net_torque_world: Vector3
    contact_count: int

    def __post_init__(self) -> None:
        _validate_aggregate_common(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_id": self.body_id,
            "center_world": list(self.center_world),
            "net_force_world": list(self.net_force_world),
            "net_torque_world": list(self.net_torque_world),
            "contact_count": self.contact_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BodyContactWrench":
        if not isinstance(data, dict):
            raise InvalidRuntimeStateError(f"body contact wrench data must be a dict; actual value={data!r}")
        return cls(
            body_id=data.get("body_id"),
            center_world=tuple(data.get("center_world", ())),
            net_force_world=tuple(data.get("net_force_world", ())),
            net_torque_world=tuple(data.get("net_torque_world", ())),
            contact_count=data.get("contact_count"),
        )


@dataclass(frozen=True)
class BodyPairContactWrench:
    """Net contact wrench exchanged between a stable ordered pair of runtime bodies."""

    body_a: str
    body_b: str
    center_a_world: Vector3
    center_b_world: Vector3
    force_on_body_a_world: Vector3
    torque_on_body_a_world: Vector3
    force_on_body_b_world: Vector3
    torque_on_body_b_world: Vector3
    contact_count: int

    def __post_init__(self) -> None:
        body_a = _non_empty_string(self.body_a, field_name="body_a", error_type=InvalidRuntimeStateError)
        body_b = _non_empty_string(self.body_b, field_name="body_b", error_type=InvalidRuntimeStateError)
        if body_a == body_b:
            raise InvalidRuntimeStateError(f"body_a and body_b must be different; actual value={body_a!r}")
        if body_b < body_a:
            raise InvalidRuntimeStateError(
                f"BodyPairContactWrench expects stable body_a <= body_b ordering; body_a={body_a!r}, body_b={body_b!r}"
            )
        object.__setattr__(self, "body_a", body_a)
        object.__setattr__(self, "body_b", body_b)
        for field_name in (
            "center_a_world",
            "center_b_world",
            "force_on_body_a_world",
            "torque_on_body_a_world",
            "force_on_body_b_world",
            "torque_on_body_b_world",
        ):
            object.__setattr__(
                self,
                field_name,
                _as_float_tuple(
                    getattr(self, field_name),
                    field_name=field_name,
                    length=3,
                    error_type=InvalidRuntimeStateError,
                ),
            )
        _validate_contact_count(self.contact_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_a": self.body_a,
            "body_b": self.body_b,
            "center_a_world": list(self.center_a_world),
            "center_b_world": list(self.center_b_world),
            "force_on_body_a_world": list(self.force_on_body_a_world),
            "torque_on_body_a_world": list(self.torque_on_body_a_world),
            "force_on_body_b_world": list(self.force_on_body_b_world),
            "torque_on_body_b_world": list(self.torque_on_body_b_world),
            "contact_count": self.contact_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BodyPairContactWrench":
        if not isinstance(data, dict):
            raise InvalidRuntimeStateError(f"body-pair contact wrench data must be a dict; actual value={data!r}")
        return cls(
            body_a=data.get("body_a"),
            body_b=data.get("body_b"),
            center_a_world=tuple(data.get("center_a_world", ())),
            center_b_world=tuple(data.get("center_b_world", ())),
            force_on_body_a_world=tuple(data.get("force_on_body_a_world", ())),
            torque_on_body_a_world=tuple(data.get("torque_on_body_a_world", ())),
            force_on_body_b_world=tuple(data.get("force_on_body_b_world", ())),
            torque_on_body_b_world=tuple(data.get("torque_on_body_b_world", ())),
            contact_count=data.get("contact_count"),
        )


@dataclass(frozen=True)
class BodyContactImpulse:
    """Discrete time-integrated contact impulse for one runtime body."""

    body_id: str
    linear_impulse_world: Vector3
    angular_impulse_world: Vector3
    duration: float
    sample_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "body_id",
            _non_empty_string(self.body_id, field_name="body_id", error_type=InvalidRuntimeStateError),
        )
        for field_name in ("linear_impulse_world", "angular_impulse_world"):
            object.__setattr__(
                self,
                field_name,
                _as_float_tuple(
                    getattr(self, field_name),
                    field_name=field_name,
                    length=3,
                    error_type=InvalidRuntimeStateError,
                ),
            )
        object.__setattr__(
            self,
            "duration",
            _finite_float(
                self.duration,
                field_name="duration",
                minimum=0.0,
                error_type=InvalidRuntimeStateError,
            ),
        )
        _validate_contact_count(self.sample_count, field_name="sample_count", allow_zero=True)


def force_on_body(wrench: ContactWrench, runtime_body_id: str) -> Vector3:
    """Return the force that one contact wrench applies to a runtime body."""
    if wrench.contact.body_a == runtime_body_id:
        return wrench.force_on_body_a_world
    if wrench.contact.body_b == runtime_body_id:
        return wrench.force_on_body_b_world
    raise InvalidRuntimeStateError(
        f"wrench does not involve runtime body; runtime_body_id={runtime_body_id!r}, "
        f"body_a={wrench.contact.body_a!r}, body_b={wrench.contact.body_b!r}"
    )


def pure_contact_torque_on_body(wrench: ContactWrench, runtime_body_id: str) -> Vector3:
    """Return pure solver contact torque for one runtime body."""
    if wrench.contact.body_a == runtime_body_id:
        return wrench.contact_torque_on_body_a_world
    if wrench.contact.body_b == runtime_body_id:
        return wrench.contact_torque_on_body_b_world
    raise InvalidRuntimeStateError(
        f"wrench does not involve runtime body; runtime_body_id={runtime_body_id!r}, "
        f"body_a={wrench.contact.body_a!r}, body_b={wrench.contact.body_b!r}"
    )


def torque_about_center_from_wrench(
    wrench: ContactWrench,
    *,
    runtime_body_id: str,
    center_world: Vector3,
) -> Vector3:
    """Return total contact torque about ``center_world`` for one body."""
    center = _vector3(center_world, field_name="center_world")
    force = force_on_body(wrench, runtime_body_id)
    pure_torque = pure_contact_torque_on_body(wrench, runtime_body_id)
    lever = _subtract(wrench.contact.position, center)
    return _add(pure_torque, _cross(lever, force))


def aggregate_contact_wrenches_by_body(
    wrenches: Iterable[ContactWrench],
    centers_world: Mapping[str, Vector3],
) -> tuple[BodyContactWrench, ...]:
    """Aggregate contact wrenches into net body wrenches about supplied centers."""
    centers = {body_id: _vector3(center, field_name=f"centers_world[{body_id!r}]") for body_id, center in centers_world.items()}
    accum: dict[str, tuple[Vector3, Vector3, int]] = {}
    for wrench in wrenches:
        for body_id in (wrench.contact.body_a, wrench.contact.body_b):
            if body_id not in centers:
                raise InvalidRuntimeStateError(f"missing center for runtime body; body_id={body_id!r}")
            net_force, net_torque, count = accum.get(body_id, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0))
            accum[body_id] = (
                _add(net_force, force_on_body(wrench, body_id)),
                _add(
                    net_torque,
                    torque_about_center_from_wrench(
                        wrench,
                        runtime_body_id=body_id,
                        center_world=centers[body_id],
                    ),
                ),
                count + 1,
            )
    return tuple(
        BodyContactWrench(
            body_id=body_id,
            center_world=centers[body_id],
            net_force_world=values[0],
            net_torque_world=values[1],
            contact_count=values[2],
        )
        for body_id, values in sorted(accum.items())
    )


def aggregate_contact_wrenches_by_body_pair(
    wrenches: Iterable[ContactWrench],
    centers_world: Mapping[str, Vector3],
) -> tuple[BodyPairContactWrench, ...]:
    """Aggregate contact wrenches into stable ordered body-pair wrenches."""
    centers = {body_id: _vector3(center, field_name=f"centers_world[{body_id!r}]") for body_id, center in centers_world.items()}
    accum: dict[tuple[str, str], tuple[Vector3, Vector3, Vector3, Vector3, int]] = {}
    for wrench in wrenches:
        body_a, body_b = sorted((wrench.contact.body_a, wrench.contact.body_b))
        if body_a not in centers or body_b not in centers:
            missing = body_a if body_a not in centers else body_b
            raise InvalidRuntimeStateError(f"missing center for runtime body; body_id={missing!r}")
        force_a = force_on_body(wrench, body_a)
        force_b = force_on_body(wrench, body_b)
        torque_a = torque_about_center_from_wrench(wrench, runtime_body_id=body_a, center_world=centers[body_a])
        torque_b = torque_about_center_from_wrench(wrench, runtime_body_id=body_b, center_world=centers[body_b])
        old_force_a, old_torque_a, old_force_b, old_torque_b, count = accum.get(
            (body_a, body_b),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0),
        )
        accum[(body_a, body_b)] = (
            _add(old_force_a, force_a),
            _add(old_torque_a, torque_a),
            _add(old_force_b, force_b),
            _add(old_torque_b, torque_b),
            count + 1,
        )
    return tuple(
        BodyPairContactWrench(
            body_a=body_a,
            body_b=body_b,
            center_a_world=centers[body_a],
            center_b_world=centers[body_b],
            force_on_body_a_world=values[0],
            torque_on_body_a_world=values[1],
            force_on_body_b_world=values[2],
            torque_on_body_b_world=values[3],
            contact_count=values[4],
        )
        for (body_a, body_b), values in sorted(accum.items())
    )


def integrate_body_contact_impulse(
    samples: Iterable[BodyContactWrench],
    *,
    timestep: float,
    body_id: str | None = None,
) -> BodyContactImpulse:
    """Integrate body contact wrenches with a fixed timestep using rectangle rule."""
    dt = _finite_float(
        timestep,
        field_name="timestep",
        minimum=0.0,
        strict_minimum=True,
        error_type=InvalidRuntimeStateError,
    )
    sample_tuple = tuple(samples)
    if not sample_tuple:
        if body_id is None:
            raise InvalidRuntimeStateError("body_id is required when integrating empty contact impulse samples")
        return BodyContactImpulse(
            body_id=body_id,
            linear_impulse_world=(0.0, 0.0, 0.0),
            angular_impulse_world=(0.0, 0.0, 0.0),
            duration=0.0,
            sample_count=0,
        )
    selected_body = body_id or sample_tuple[0].body_id
    linear = (0.0, 0.0, 0.0)
    angular = (0.0, 0.0, 0.0)
    count = 0
    for sample in sample_tuple:
        if sample.body_id != selected_body:
            continue
        linear = _add(linear, _scale(sample.net_force_world, dt))
        angular = _add(angular, _scale(sample.net_torque_world, dt))
        count += 1
    return BodyContactImpulse(
        body_id=selected_body,
        linear_impulse_world=linear,
        angular_impulse_world=angular,
        duration=count * dt,
        sample_count=count,
    )


def _validate_aggregate_common(value: BodyContactWrench) -> None:
    object.__setattr__(
        value,
        "body_id",
        _non_empty_string(value.body_id, field_name="body_id", error_type=InvalidRuntimeStateError),
    )
    for field_name in ("center_world", "net_force_world", "net_torque_world"):
        object.__setattr__(value, field_name, _vector3(getattr(value, field_name), field_name=field_name))
    _validate_contact_count(value.contact_count)


def _validate_contact_count(value: Any, *, field_name: str = "contact_count", allow_zero: bool = False) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < (0 if allow_zero else 1):
        minimum = ">= 0" if allow_zero else ">= 1"
        raise InvalidRuntimeStateError(f"{field_name} must be an int {minimum}; actual value={value!r}")


def _vector3(value: Any, *, field_name: str) -> Vector3:
    return tuple(
        _as_float_tuple(value, field_name=field_name, length=3, error_type=InvalidRuntimeStateError)
    )  # type: ignore[return-value]


def _add(first: Vector3, second: Vector3) -> Vector3:
    return tuple(first[index] + second[index] for index in range(3))


def _subtract(first: Vector3, second: Vector3) -> Vector3:
    return tuple(first[index] - second[index] for index in range(3))


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return tuple(value * scalar for value in vector)


def _cross(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
