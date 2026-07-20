"""Demonstrate wrapping a dynamic body as a reusable PhysicsAssetSpec."""

from physical_simulation.assets import PhysicsAssetSpec, create_box, create_single_body_asset


def main() -> None:
    """Run the physics asset example."""
    body = create_box(
        body_id="crate_body",
        size=(0.6, 0.4, 0.4),
        mass=2.0,
        name="crate_body",
    )
    asset = create_single_body_asset(
        asset_id="crate_asset",
        body=body,
        name="Wooden Crate",
        metadata={"source": "parametric_builder", "category": "container"},
    )
    print(asset.to_json())
    restored = PhysicsAssetSpec.from_json(asset.to_json())
    print(f"round_trip_equal: {restored == asset}")


if __name__ == "__main__":
    main()
