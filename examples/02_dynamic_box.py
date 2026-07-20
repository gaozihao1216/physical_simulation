"""Create, save, load, and compare a dynamic box asset."""

from pathlib import Path

from physical_simulation.assets import Transform, create_box
from physical_simulation.serialization import load_rigid_body, save_rigid_body


def main() -> None:
    """Run the dynamic box example."""
    box = create_box(
        body_id="box_001",
        size=(1.0, 0.5, 0.3),
        mass=2.0,
        transform=Transform(position=(0.0, 0.0, 1.0)),
    )
    assert box.mass_properties is not None
    print(f"mass: {box.mass_properties.mass}")
    print(f"inertia_diagonal: {box.mass_properties.inertia_diagonal}")

    path = Path("outputs/examples/dynamic_box.json")
    save_rigid_body(box, path)
    restored = load_rigid_body(path)
    print(f"round_trip_equal: {restored == box}")


if __name__ == "__main__":
    main()
