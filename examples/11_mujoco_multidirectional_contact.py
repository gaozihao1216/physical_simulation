"""Demonstrate multi-directional contact wrenches without a GUI."""

from __future__ import annotations

import math

from physical_simulation.assets import (
    BoxGeometry,
    ColliderSpec,
    RigidBodySpec,
    Transform,
    create_box,
    create_ground,
    create_single_body_asset,
    create_sphere,
)
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.runtime import ContactWrench, make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, create_scene


def qy(degrees: float) -> tuple[float, float, float, float]:
    radians = math.radians(degrees)
    return (math.cos(radians / 2.0), 0.0, math.sin(radians / 2.0), 0.0)


def force_on_body(wrench: ContactWrench, runtime_body_id: str) -> tuple[float, float, float]:
    if wrench.contact.body_a == runtime_body_id:
        return wrench.force_on_body_a_world
    if wrench.contact.body_b == runtime_body_id:
        return wrench.force_on_body_b_world
    raise ValueError(f"wrench does not involve {runtime_body_id!r}")


def cross(first, second):
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def add(first, second):
    return tuple(first[index] + second[index] for index in range(3))


def aggregate(wrenches, *, body_id: str, center):
    net_force = (0.0, 0.0, 0.0)
    net_torque = (0.0, 0.0, 0.0)
    for wrench in wrenches:
        if body_id not in (wrench.contact.body_a, wrench.contact.body_b):
            continue
        force = force_on_body(wrench, body_id)
        lever = tuple(wrench.contact.position[index] - center[index] for index in range(3))
        pure_torque = (
            wrench.contact_torque_on_body_a_world
            if wrench.contact.body_a == body_id
            else wrench.contact_torque_on_body_b_world
        )
        net_force = add(net_force, force)
        net_torque = add(net_torque, add(pure_torque, cross(lever, force)))
    return net_force, net_torque


def create_v_groove_scene():
    ramp_geometry = BoxGeometry((1.2, 1.0, 0.08))

    def ramp_asset(asset_id: str, body_id: str, *, angle_degrees: float, x: float, group: int):
        body = RigidBodySpec(
            body_id,
            body_id,
            "static",
            Transform(position=(x, 0.0, 0.161), rotation=qy(angle_degrees)),
            (),
            (
                ColliderSpec(
                    f"{body_id}_collider",
                    ramp_geometry,
                    collision_group=group,
                    collision_mask=1,
                ),
            ),
        )
        return create_single_body_asset(asset_id=asset_id, body=body)

    sphere_body = create_sphere("sphere_body", 0.1, mass=1.0, create_visual=False)
    sphere_collider = sphere_body.colliders[0]
    sphere_body = RigidBodySpec(
        sphere_body.body_id,
        sphere_body.name,
        sphere_body.body_type,
        sphere_body.transform,
        sphere_body.visuals,
        (
            ColliderSpec(
                sphere_collider.collider_id,
                sphere_collider.geometry,
                sphere_collider.local_transform,
                sphere_collider.material_id,
                collision_group=1,
                collision_mask=2 | 4,
            ),
        ),
        sphere_body.mass_properties,
    )
    return create_scene(
        scene_id="example_v_groove",
        instances=(
            AssetInstanceSpec("left_ramp", ramp_asset("left_asset", "left_ramp_body", angle_degrees=35.0, x=-0.3, group=2), fixed_base=True),
            AssetInstanceSpec("right_ramp", ramp_asset("right_asset", "right_ramp_body", angle_degrees=-35.0, x=0.3, group=4), fixed_base=True),
            AssetInstanceSpec("sphere", create_single_body_asset(asset_id="sphere_asset", body=sphere_body), Transform(position=(0.0, 0.0, 0.8))),
        ),
        timestep=1.0 / 240.0,
    )


def create_offcenter_box_scene():
    ground_asset = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box_asset = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.6, 0.2, 0.2), mass=1.0),
    )
    return create_scene(
        scene_id="example_offcenter_box",
        instances=(
            AssetInstanceSpec("ground", ground_asset, fixed_base=True),
            AssetInstanceSpec("box", box_asset, Transform(position=(0.0, 0.0, 1.0), rotation=qy(20.0))),
        ),
        timestep=1.0 / 240.0,
    )


def print_v_groove() -> None:
    sphere_id = make_runtime_body_id("sphere", "sphere_body")
    left_id = make_runtime_body_id("left_ramp", "left_ramp_body")
    right_id = make_runtime_body_id("right_ramp", "right_ramp_body")
    backend = MuJoCoBackend()
    backend.load_scene(create_v_groove_scene())
    backend.reset()
    for _ in range(1500):
        backend.step()
    state = backend.get_body_state(sphere_id)
    active = tuple(
        wrench
        for wrench in backend.get_contact_wrenches()
        if sphere_id in (wrench.contact.body_a, wrench.contact.body_b)
        and wrench.normal_force_magnitude > 1.0e-6
    )
    net_force, net_torque = aggregate(active, body_id=sphere_id, center=state.position)

    print("Part A: V groove")
    print(f"sphere runtime body ID = {sphere_id}")
    print(f"left support runtime body ID = {left_id}")
    print(f"right support runtime body ID = {right_id}")
    for index, wrench in enumerate(active, start=1):
        external = wrench.contact.body_a if wrench.contact.body_b == sphere_id else wrench.contact.body_b
        print(f"contact {index}: external={external}")
        print(f"  position={wrench.contact.position}")
        print(f"  force_on_sphere={force_on_body(wrench, sphere_id)}")
        print(f"  normal_force_magnitude={wrench.normal_force_magnitude:.6f}")
    print(f"sphere net contact force = {net_force}")
    print(f"sphere net torque about COM = {net_torque}")
    print("expected gravity balance = (0, 0, 9.81)")
    backend.close()


def print_offcenter_impact() -> None:
    box_id = make_runtime_body_id("box", "box_body")
    backend = MuJoCoBackend()
    backend.load_scene(create_offcenter_box_scene())
    previous = backend.reset()
    for _ in range(600):
        current = backend.step()
        active = tuple(
            wrench
            for wrench in backend.get_contact_wrenches()
            if box_id in (wrench.contact.body_a, wrench.contact.body_b)
            and wrench.normal_force_magnitude > 1.0e-6
        )
        if active:
            post = backend.step()
            state = current.get_body_state(box_id)
            previous_state = previous.get_body_state(box_id)
            post_state = post.get_body_state(box_id)
            net_force, net_torque = aggregate(active, body_id=box_id, center=state.position)

            print("\nPart B: off-center impact")
            print(f"first impact step = {current.step_index}")
            print(f"first impact time = {current.time:.6f}")
            print(f"box COM = {state.position}")
            for index, wrench in enumerate(active, start=1):
                print(f"contact {index}: position={wrench.contact.position}")
                print(f"  force_on_box={force_on_body(wrench, box_id)}")
            print(f"net force on box = {net_force}")
            print(f"net torque about COM = {net_torque}")
            print(f"angular velocity before = {previous_state.angular_velocity}")
            print(f"angular velocity after = {post_state.angular_velocity}")
            backend.close()
            return
        previous = current
    backend.close()
    raise RuntimeError("no off-center impact found")


def main() -> None:
    print_v_groove()
    print_offcenter_impact()


if __name__ == "__main__":
    main()
