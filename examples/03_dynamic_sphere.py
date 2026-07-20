"""Create a dynamic sphere using density-derived mass."""

from physical_simulation.assets import create_sphere


def main() -> None:
    """Run the dynamic sphere example."""
    sphere = create_sphere(body_id="sphere_001", radius=0.25, density=500.0)
    assert sphere.mass_properties is not None
    print(f"volume: {sphere.colliders[0].geometry.volume()}")
    print(f"mass: {sphere.mass_properties.mass}")
    print(sphere.to_json())


if __name__ == "__main__":
    main()
