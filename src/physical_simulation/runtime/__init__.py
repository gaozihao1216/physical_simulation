"""Runtime state data structures.

These structures describe runtime outputs only. They do not implement a
simulation loop or backend behavior.
"""

from physical_simulation.runtime.body_state import RigidBodyState
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
    "SimulationStepResult",
    "make_runtime_body_id",
    "make_runtime_joint_id",
]
