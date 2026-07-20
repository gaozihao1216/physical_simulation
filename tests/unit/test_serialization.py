import pytest

from physical_simulation.assets import RigidBodySpec, create_box
from physical_simulation.serialization import from_json_rigid_body, load_rigid_body, save_rigid_body, to_json
from physical_simulation.validation.errors import InvalidGeometryError, SerializationError


def test_save_and_load_rigid_body(tmp_path) -> None:
    body = create_box("box", (1.0, 1.0, 1.0), mass=1.0)
    path = tmp_path / "box.json"
    save_rigid_body(body, path)
    assert load_rigid_body(path) == body


def test_invalid_json_raises_serialization_error() -> None:
    with pytest.raises(SerializationError, match="invalid"):
        from_json_rigid_body("{")


def test_unknown_geometry_type_raises() -> None:
    text = """
    {
      "body_id": "body",
      "name": "body",
      "body_type": "static",
      "transform": {},
      "visuals": [{"visual_id": "v", "geometry": {"shape_type": "mesh"}}],
      "colliders": [],
      "mass_properties": null
    }
    """
    with pytest.raises(InvalidGeometryError, match="shape_type"):
        from_json_rigid_body(text)


def test_round_trip_keeps_object_equal() -> None:
    body = create_box("box", (1.0, 1.0, 1.0), mass=1.0)
    assert RigidBodySpec.from_json(to_json(body)) == body
