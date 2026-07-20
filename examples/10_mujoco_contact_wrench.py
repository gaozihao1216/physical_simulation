"""Print MuJoCo contact wrenches for a 1 kg box settling on ground."""

from __future__ import annotations

from physical_simulation.assets import Transform, create_box, create_ground, create_single_body_asset
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def main() -> None:
    ground_asset = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box_asset = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.4, 0.4, 0.4), mass=1.0),
    )
    scene = create_scene(
        scene_id="contact_wrench_example",
        instances=(
            AssetInstanceSpec("ground_01", ground_asset, fixed_base=True),
            AssetInstanceSpec("box_01", box_asset, Transform(position=(0.0, 0.0, 1.0))),
        ),
        gravity=(0.0, 0.0, -9.81),
        timestep=1.0 / 240.0,
    )
    box_id = make_runtime_body_id("box_01", "box_body")
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    backend.reset()
    for _ in range(720):
        backend.step()

    box_force = [0.0, 0.0, 0.0]
    wrenches = backend.get_contact_wrenches()
    for index, wrench in enumerate(wrenches, start=1):
        print(f"contact {index}: {wrench.contact.body_a} <-> {wrench.contact.body_b}")
        print(f"  position={wrench.contact.position}")
        print(f"  normal={wrench.contact.normal}")
        print(f"  force_on_body_a_world={wrench.force_on_body_a_world}")
        print(f"  force_on_body_b_world={wrench.force_on_body_b_world}")
        print(f"  normal_force_magnitude={wrench.normal_force_magnitude:.6f}")
        print(f"  tangential_force_magnitude={wrench.tangential_force_magnitude:.6f}")
        if wrench.contact.body_a == box_id:
            force_on_box = wrench.force_on_body_a_world
        elif wrench.contact.body_b == box_id:
            force_on_box = wrench.force_on_body_b_world
        else:
            continue
        for axis in range(3):
            box_force[axis] += force_on_box[axis]

    print("expected support force = 9.81 N")
    print(f"measured support force = {box_force[2]:.6f} N")
    backend.close()


if __name__ == "__main__":
    main()
