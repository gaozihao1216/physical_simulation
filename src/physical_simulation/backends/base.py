"""Abstract interface for physics backend adapters.

TODO: Extend type definitions alongside the Physics IR.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class PhysicsBackend(ABC):
    """Minimal abstract interface for simulation backends."""

    @abstractmethod
    def load_scene(self, scene: Any) -> None:
        """Load a scene into the backend."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset backend state."""

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance simulation by one time step."""

    @abstractmethod
    def get_body_state(self, body_id: str) -> Any:
        """Return state for a body."""

    @abstractmethod
    def get_joint_state(self, joint_id: str) -> Any:
        """Return state for a joint."""

    @abstractmethod
    def get_contacts(self) -> Any:
        """Return current contact information."""

    @abstractmethod
    def apply_force(self, body_id: str, force: Any, point: Optional[Any] = None) -> None:
        """Apply a force to a body."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""
