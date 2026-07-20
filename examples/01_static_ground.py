"""Create a static finite ground body and print its JSON representation."""

from physical_simulation.assets import create_ground


def main() -> None:
    """Run the static ground example."""
    ground = create_ground()
    print(ground)
    print(ground.to_json())


if __name__ == "__main__":
    main()
