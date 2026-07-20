"""Create a static table body composed from multiple visuals and colliders."""

from physical_simulation.assets import (
    BoxGeometry,
    ColliderSpec,
    RigidBodySpec,
    Transform,
    VisualSpec,
)


def main() -> None:
    """Run the compound table example."""
    tabletop = BoxGeometry(size=(1.6, 0.9, 0.08))
    leg = BoxGeometry(size=(0.08, 0.08, 0.7))
    leg_positions = (
        (-0.7, -0.35, -0.39),
        (0.7, -0.35, -0.39),
        (-0.7, 0.35, -0.39),
        (0.7, 0.35, -0.39),
    )

    visuals = [
        VisualSpec("tabletop_visual", tabletop),
    ]
    colliders = [
        ColliderSpec("tabletop_collider", tabletop),
    ]
    for index, position in enumerate(leg_positions, start=1):
        transform = Transform(position=position)
        visuals.append(VisualSpec(f"leg_{index}_visual", leg, local_transform=transform))
        colliders.append(ColliderSpec(f"leg_{index}_collider", leg, local_transform=transform))

    table = RigidBodySpec(
        body_id="table_001",
        name="compound_table",
        body_type="static",
        transform=Transform.identity(),
        visuals=tuple(visuals),
        colliders=tuple(colliders),
    )
    print(table.to_json())


if __name__ == "__main__":
    main()
