"""Abstract interface for physics backend adapters.

TODO: Extend type definitions alongside the Physics IR.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from physical_simulation.runtime import ContactPoint, ContactWrench, RigidBodyState, SimulationStepResult


class PhysicsBackend(ABC):
    """Minimal abstract interface for simulation backends."""

    @abstractmethod
    def load_scene(self, scene: Any) -> None:
        """Load a scene into the backend."""

    @abstractmethod
    def reset(self) -> SimulationStepResult:
        """Reset backend state."""

    @abstractmethod
    def step(self, action: object | None = None) -> SimulationStepResult:
        """Advance simulation by one time step."""

    @abstractmethod
    def get_body_state(self, runtime_body_id: str) -> RigidBodyState:
        """Return state for a body."""

    @abstractmethod
    def get_joint_state(self, joint_id: str) -> Any:
        """Return state for a joint."""

    @abstractmethod
    def get_contacts(self) -> tuple[ContactPoint, ...]:
        """Return current contact information."""

    @abstractmethod
    def get_contact_wrenches(self) -> tuple[ContactWrench, ...]:
        """Return current contact wrenches."""

    @abstractmethod
    def apply_force(self, body_id: str, force: Any, point: Optional[Any] = None) -> None:
        """Apply a force to a body."""

    @abstractmethod
    def apply_torque(self, body_id: str, torque: Any) -> None:
        """Apply a torque to a body."""

    @abstractmethod
    def clear_applied_forces(self) -> None:
        """Clear externally applied forces and torques."""

    @abstractmethod
    def set_body_velocity(
        self,
        body_id: str,
        linear_velocity: Any,
        angular_velocity: Any = (0.0, 0.0, 0.0),
        *,
        update_initial: bool = False,
    ) -> SimulationStepResult:
        """Set a free body's velocity."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""
