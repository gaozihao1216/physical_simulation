import pytest

from physical_simulation.assets import DEFAULT_MATERIAL, PhysicsMaterialSpec
from physical_simulation.validation.errors import PhysicsValidationError


def test_default_material_values() -> None:
    assert DEFAULT_MATERIAL.material_id == "default"
    assert DEFAULT_MATERIAL.density == 1000.0


def test_invalid_friction_raises() -> None:
    with pytest.raises(PhysicsValidationError, match="static_friction"):
        PhysicsMaterialSpec("bad", static_friction=-0.1)


def test_invalid_restitution_raises() -> None:
    with pytest.raises(PhysicsValidationError, match="restitution"):
        PhysicsMaterialSpec("bad", restitution=1.5)


def test_invalid_density_raises() -> None:
    with pytest.raises(PhysicsValidationError, match="density"):
        PhysicsMaterialSpec("bad", density=0.0)


def test_dict_round_trip() -> None:
    material = PhysicsMaterialSpec("wood", static_friction=0.6, dynamic_friction=0.5, density=700.0)
    assert PhysicsMaterialSpec.from_dict(material.to_dict()) == material
