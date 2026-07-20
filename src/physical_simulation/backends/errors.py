"""Backend-specific errors."""


class PhysicsBackendError(RuntimeError):
    """Base error for physics backend failures."""


class BackendNotLoadedError(PhysicsBackendError):
    """Raised when an operation requires a loaded scene."""


class MuJoCoUnavailableError(PhysicsBackendError, ImportError):
    """Raised when the optional MuJoCo dependency is unavailable."""


class MuJoCoModelLoadingError(PhysicsBackendError):
    """Raised when MJCF cannot be loaded into a MuJoCo model."""


class MuJoCoRuntimeError(PhysicsBackendError):
    """Raised when MuJoCo simulation or state extraction fails."""


class UnsupportedBackendOperationError(PhysicsBackendError):
    """Raised when an operation is not supported in the current phase."""


class UnknownRuntimeBodyError(PhysicsBackendError, KeyError):
    """Raised when a runtime body ID is not present in the loaded model."""


class UnknownRuntimeGeomError(PhysicsBackendError, KeyError):
    """Raised when a compiled collision geom cannot be found."""
