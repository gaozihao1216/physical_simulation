import pytest

from physical_simulation.assets import (
    DEFAULT_MATERIAL,
    ColliderSpec,
    PhysicsAssetSpec,
    PhysicsMaterialSpec,
    create_box,
    create_single_body_asset,
)
from physical_simulation.validation.errors import InvalidPhysicsAssetError


def test_single_body_asset_creation_and_default_material() -> None:
    body = create_box("box_body", (1.0, 1.0, 1.0), mass=1.0)
    asset = create_single_body_asset(asset_id="box_asset", body=body)
    assert asset.name == body.name
    assert asset.materials == (DEFAULT_MATERIAL,)
    assert asset.bodies == (body,)


def test_asset_validation_errors() -> None:
    body = create_box("box_body", (1.0, 1.0, 1.0), mass=1.0)
    with pytest.raises(InvalidPhysicsAssetError, match="asset_id"):
        create_single_body_asset(asset_id="", body=body)
    with pytest.raises(InvalidPhysicsAssetError, match="name"):
        PhysicsAssetSpec("1.0", "a", "", (DEFAULT_MATERIAL,), (body,))
    with pytest.raises(InvalidPhysicsAssetError, match="schema_version"):
        PhysicsAssetSpec("2.0", "a", "a", (DEFAULT_MATERIAL,), (body,))
    with pytest.raises(InvalidPhysicsAssetError, match="bodies"):
        PhysicsAssetSpec("1.0", "a", "a", (DEFAULT_MATERIAL,), ())


def test_duplicate_ids_and_missing_materials_raise() -> None:
    body = create_box("box_body", (1.0, 1.0, 1.0), mass=1.0)
    with pytest.raises(InvalidPhysicsAssetError, match="body_id"):
        PhysicsAssetSpec("1.0", "a", "a", (DEFAULT_MATERIAL,), (body, body))
    with pytest.raises(InvalidPhysicsAssetError, match="material_id"):
        PhysicsAssetSpec("1.0", "a", "a", (DEFAULT_MATERIAL, DEFAULT_MATERIAL), (body,))
    custom_body = create_box(
        "custom_body",
        (1.0, 1.0, 1.0),
        mass=1.0,
        material=PhysicsMaterialSpec("rubber", density=1100.0),
    )
    with pytest.raises(InvalidPhysicsAssetError, match="collider.material_id"):
        create_single_body_asset(asset_id="custom", body=custom_body)


def test_metadata_is_copied_and_read_only() -> None:
    body = create_box("box_body", (1.0, 1.0, 1.0), mass=1.0)
    metadata = {"source": "test"}
    asset = create_single_body_asset(asset_id="box_asset", body=body, metadata=metadata)
    metadata["source"] = "changed"
    assert asset.metadata["source"] == "test"
    with pytest.raises(TypeError):
        asset.metadata["source"] = "nope"  # type: ignore[index]
    with pytest.raises(InvalidPhysicsAssetError, match="metadata"):
        create_single_body_asset(asset_id="bad", body=body, metadata={"ok": 1})  # type: ignore[arg-type]


def test_asset_round_trip() -> None:
    body = create_box("box_body", (1.0, 1.0, 1.0), mass=1.0)
    asset = create_single_body_asset(asset_id="box_asset", body=body, metadata={"category": "box"})
    assert PhysicsAssetSpec.from_dict(asset.to_dict()) == asset
    assert PhysicsAssetSpec.from_json(asset.to_json()) == asset
