"""Runtime state data structures.

These structures describe runtime outputs only. They do not implement a
simulation loop or backend behavior.
"""

from physical_simulation.runtime.body_state import RigidBodyState
from physical_simulation.runtime.contact_aggregation import (
    BodyContactImpulse,
    BodyContactWrench,
    BodyPairContactWrench,
    aggregate_contact_wrenches_by_body,
    aggregate_contact_wrenches_by_body_pair,
    force_on_body,
    integrate_body_contact_impulse,
    pure_contact_torque_on_body,
    torque_about_center_from_wrench,
)
from physical_simulation.runtime.contact import ContactPoint
from physical_simulation.runtime.contact_wrench import ContactWrench
from physical_simulation.runtime.ids import make_runtime_body_id, make_runtime_joint_id
from physical_simulation.runtime.joint_state import JointState
from physical_simulation.runtime.step_result import SimulationStepResult

__all__ = [
    "RigidBodyState",
    "JointState",
    "ContactPoint",
    "ContactWrench",
    "BodyContactWrench",
    "BodyPairContactWrench",
    "BodyContactImpulse",
    "force_on_body",
    "pure_contact_torque_on_body",
    "torque_about_center_from_wrench",
    "aggregate_contact_wrenches_by_body",
    "aggregate_contact_wrenches_by_body_pair",
    "integrate_body_contact_impulse",
    "SimulationStepResult",
    "make_runtime_body_id",
    "make_runtime_joint_id",
]
