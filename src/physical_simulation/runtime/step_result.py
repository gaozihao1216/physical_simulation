"""Runtime simulation step result data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from physical_simulation.runtime.body_state import RigidBodyState
from physical_simulation.runtime.contact import ContactPoint
from physical_simulation.runtime.joint_state import JointState
from physical_simulation.validation.asset_validator import _finite_float
from physical_simulation.validation.errors import InvalidRuntimeStateError


@dataclass(frozen=True)
class SimulationStepResult:
    """Runtime output for one simulation step."""

    time: float
    step_index: int
    body_states: tuple[RigidBodyState, ...]
    joint_states: tuple[JointState, ...] = ()
    contacts: tuple[ContactPoint, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "time",
            _finite_float(
                self.time,
                field_name="time",
                minimum=0.0,
                error_type=InvalidRuntimeStateError,
            ),
        )
        if not isinstance(self.step_index, int) or isinstance(self.step_index, bool) or self.step_index < 0:
            raise InvalidRuntimeStateError(
                f"step_index must be a non-negative int; actual value={self.step_index!r}"
            )
        body_states = tuple(self.body_states)
        joint_states = tuple(self.joint_states)
        contacts = tuple(self.contacts)
        for state in body_states:
            if not isinstance(state, RigidBodyState):
                raise InvalidRuntimeStateError(
                    f"body_states must contain RigidBodyState values; actual value={state!r}"
                )
        for state in joint_states:
            if not isinstance(state, JointState):
                raise InvalidRuntimeStateError(
                    f"joint_states must contain JointState values; actual value={state!r}"
                )
        for contact in contacts:
            if not isinstance(contact, ContactPoint):
                raise InvalidRuntimeStateError(
                    f"contacts must contain ContactPoint values; actual value={contact!r}"
                )
        body_ids = [state.body_id for state in body_states]
        joint_ids = [state.joint_id for state in joint_states]
        if len(body_ids) != len(set(body_ids)):
            raise InvalidRuntimeStateError(f"body_id values must be unique; actual IDs={body_ids!r}")
        if len(joint_ids) != len(set(joint_ids)):
            raise InvalidRuntimeStateError(f"joint_id values must be unique; actual IDs={joint_ids!r}")
        object.__setattr__(self, "body_states", body_states)
        object.__setattr__(self, "joint_states", joint_states)
        object.__setattr__(self, "contacts", contacts)

    @property
    def has_contacts(self) -> bool:
        """Return whether this step contains contacts."""
        return bool(self.contacts)

    def get_body_state(self, body_id: str) -> RigidBodyState:
        """Return state for a body id, or raise KeyError."""
        for state in self.body_states:
            if state.body_id == body_id:
                return state
        raise KeyError(f"body_id not found in SimulationStepResult: {body_id!r}")

    def get_joint_state(self, joint_id: str) -> JointState:
        """Return state for a joint id, or raise KeyError."""
        for state in self.joint_states:
            if state.joint_id == joint_id:
                return state
        raise KeyError(f"joint_id not found in SimulationStepResult: {joint_id!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the step result to a JSON-compatible dictionary."""
        return {
            "time": self.time,
            "step_index": self.step_index,
            "body_states": [state.to_dict() for state in self.body_states],
            "joint_states": [state.to_dict() for state in self.joint_states],
            "contacts": [contact.to_dict() for contact in self.contacts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationStepResult":
        """Deserialize a step result from a dictionary."""
        if not isinstance(data, dict):
            raise InvalidRuntimeStateError(f"step result data must be a dict; actual value={data!r}")
        return cls(
            time=data.get("time"),
            step_index=data.get("step_index"),
            body_states=tuple(RigidBodyState.from_dict(item) for item in data.get("body_states", ())),
            joint_states=tuple(JointState.from_dict(item) for item in data.get("joint_states", ())),
            contacts=tuple(ContactPoint.from_dict(item) for item in data.get("contacts", ())),
        )
