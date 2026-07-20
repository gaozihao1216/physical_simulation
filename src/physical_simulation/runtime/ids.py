"""Stable runtime identifier helpers."""

from physical_simulation.validation.asset_validator import _non_empty_string
from physical_simulation.validation.errors import InvalidRuntimeStateError

RUNTIME_ID_SEPARATOR = "/"


def _validate_runtime_id_part(value: str, field_name: str) -> str:
    value = _non_empty_string(
        value,
        field_name=field_name,
        error_type=InvalidRuntimeStateError,
    )
    if RUNTIME_ID_SEPARATOR in value:
        raise InvalidRuntimeStateError(
            f"{field_name} must not contain separator {RUNTIME_ID_SEPARATOR!r}; actual value={value!r}"
        )
    return value


def make_runtime_body_id(instance_id: str, body_id: str) -> str:
    """Return a backend-independent runtime body id as ``instance_id/body_id``."""
    return (
        f"{_validate_runtime_id_part(instance_id, 'instance_id')}"
        f"{RUNTIME_ID_SEPARATOR}"
        f"{_validate_runtime_id_part(body_id, 'body_id')}"
    )


def make_runtime_joint_id(instance_id: str, joint_id: str) -> str:
    """Return a backend-independent runtime joint id as ``instance_id/joint_id``."""
    return (
        f"{_validate_runtime_id_part(instance_id, 'instance_id')}"
        f"{RUNTIME_ID_SEPARATOR}"
        f"{_validate_runtime_id_part(joint_id, 'joint_id')}"
    )
