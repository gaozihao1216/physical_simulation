"""Project-specific validation and serialization errors."""


class PhysicsValidationError(ValueError):
    """Base error for invalid physics asset specifications."""


class InvalidGeometryError(PhysicsValidationError):
    """Raised when geometry data is invalid."""


class InvalidMassPropertiesError(PhysicsValidationError):
    """Raised when mass or inertia data is invalid."""


class InvalidRigidBodyError(PhysicsValidationError):
    """Raised when rigid body data is invalid."""


class SerializationError(PhysicsValidationError):
    """Raised when JSON serialization or deserialization fails."""
