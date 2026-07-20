"""Project-specific validation and serialization errors."""


class PhysicsValidationError(ValueError):
    """Base error for invalid physics asset specifications."""


class InvalidGeometryError(PhysicsValidationError):
    """Raised when geometry data is invalid."""


class InvalidMassPropertiesError(PhysicsValidationError):
    """Raised when mass or inertia data is invalid."""


class InvalidRigidBodyError(PhysicsValidationError):
    """Raised when rigid body data is invalid."""


class InvalidPhysicsAssetError(PhysicsValidationError):
    """Raised when a reusable physics asset specification is invalid."""


class InvalidPhysicsSceneError(PhysicsValidationError):
    """Raised when a physics scene specification is invalid."""


class InvalidRuntimeStateError(PhysicsValidationError):
    """Raised when runtime state data is invalid."""


class ScaleBakingError(PhysicsValidationError):
    """Raised when transform scale cannot be baked into geometry exactly."""


class SerializationError(PhysicsValidationError):
    """Raised when JSON serialization or deserialization fails."""
