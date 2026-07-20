"""Demonstrate runtime state data without running a physics simulation."""

from physical_simulation.runtime import (
    ContactPoint,
    RigidBodyState,
    SimulationStepResult,
    make_runtime_body_id,
)


def main() -> None:
    """Run the runtime state data example."""
    box_id = make_runtime_body_id("box_01", "box_body")
    ground_id = make_runtime_body_id("ground_01", "ground_body")
    box_state = RigidBodyState(
        body_id=box_id,
        position=(0.0, 0.0, 0.8),
        rotation=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, -1.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )
    ground_state = RigidBodyState(
        body_id=ground_id,
        position=(0.0, 0.0, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )
    contact = ContactPoint(
        body_a=box_id,
        body_b=ground_id,
        position=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, -2.0),
        penetration_depth=0.001,
    )
    result = SimulationStepResult(
        time=0.1,
        step_index=24,
        body_states=(box_state, ground_state),
        contacts=(contact,),
    )
    print(result.get_body_state(box_id))
    print(f"has_contacts: {result.has_contacts}")
    print(result.to_dict())
    restored = SimulationStepResult.from_dict(result.to_dict())
    print(f"round_trip_equal: {restored == result}")
    print("This is runtime state data only, not a real physics simulation.")


if __name__ == "__main__":
    main()
