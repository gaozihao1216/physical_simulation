"""JSON codec helpers for Physics IR assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from physical_simulation.assets.rigid_body import RigidBodySpec
from physical_simulation.validation.errors import SerializationError


def to_json(value: Any) -> str:
    """Serialize a supported Physics IR value to stable, readable JSON."""
    if hasattr(value, "to_dict"):
        data = value.to_dict()
    else:
        raise SerializationError(
            f"value must provide to_dict() for JSON serialization; actual type={type(value).__name__}"
        )
    return json.dumps(data, indent=2, sort_keys=True)


def from_json_rigid_body(text: str) -> RigidBodySpec:
    """Deserialize a rigid body from JSON text."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SerializationError(f"invalid rigid body JSON: {exc.msg} at position {exc.pos}") from exc
    return RigidBodySpec.from_dict(data)


def save_rigid_body(body: RigidBodySpec, path: Union[str, Path]) -> None:
    """Save a rigid body JSON document to disk."""
    if not isinstance(body, RigidBodySpec):
        raise SerializationError(f"body must be RigidBodySpec; actual value={body!r}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_json(body), encoding="utf-8")


def load_rigid_body(path: Union[str, Path]) -> RigidBodySpec:
    """Load a rigid body JSON document from disk."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise SerializationError(f"could not read rigid body JSON from path={source!s}: {exc}") from exc
    return from_json_rigid_body(text)
