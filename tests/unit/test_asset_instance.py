import pytest

from physical_simulation.assets import PhysicsAssetSpec, Transform, create_box, create_single_body_asset
from physical_simulation.scene import AssetInstanceSpec
from physical_simulation.validation.errors import InvalidPhysicsAssetError, InvalidPhysicsSceneError


def test_asset_instance_creation_round_trip_and_fixed_base() -> None:
    asset = create_single_body_asset(asset_id="box_asset", body=create_box("box_body", (1.0, 1.0, 1.0), mass=1.0))
    instance = AssetInstanceSpec("box_01", asset, Transform(position=(0.0, 0.0, 1.0)), fixed_base=True)
    assert instance.fixed_base is True
    assert AssetInstanceSpec.from_dict(instance.to_dict()) == instance


def test_asset_instance_validation_errors() -> None:
    asset = create_single_body_asset(asset_id="box_asset", body=create_box("box_body", (1.0, 1.0, 1.0), mass=1.0))
    with pytest.raises(InvalidPhysicsSceneError, match="instance_id"):
        AssetInstanceSpec("", asset)
    with pytest.raises(InvalidPhysicsSceneError, match="unit scale"):
        AssetInstanceSpec("box_01", asset, Transform(scale=(2.0, 2.0, 2.0)))
    with pytest.raises(InvalidPhysicsAssetError):
        bad_asset = PhysicsAssetSpec("1.0", "bad", "bad", (), ())
        AssetInstanceSpec("bad", bad_asset)
