"""Compiler-specific errors."""


class PhysicsCompilationError(RuntimeError):
    """Base error for backend compilation failures."""


class UnsupportedPhysicsFeatureError(PhysicsCompilationError):
    """Raised when the Physics IR uses an unsupported feature."""


class UnsupportedAssetStructureError(PhysicsCompilationError):
    """Raised when an asset structure cannot yet be compiled."""


class MuJoCoCompilationError(PhysicsCompilationError):
    """Raised when MJCF generation fails."""
