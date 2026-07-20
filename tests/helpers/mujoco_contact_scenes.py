"""MuJoCo integration-test scenes for contact wrench validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
from physical_simulation.runtime import ContactWrench, RigidBodyState, make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, PhysicsSceneSpec, create_scene

from tests.helpers.contact_wrench_math import Vector3

ACTIVE_FORCE_THRESHOLD = 1.0e-6
SPHERE_BODY_ID = make_runtime_body_id("sphere", "sphere_body")
LEFT_RAMP_BODY_ID = make_runtime_body_id("left_ramp", "left_ramp_body")
RIGHT_RAMP_BODY_ID = make_runtime_body_id("right_ramp", "right_ramp_body")
BOX_BODY_ID = make_runtime_body_id("box", "box_body")


@dataclass(frozen=True)
class ImpactSnapshot:
    step_index: int
    time: float
    previous_state: RigidBodyState
    impact_state: RigidBodyState
    post_impact_state: RigidBodyState
    active_wrenches: tuple[ContactWrench, ...]


def qy(degrees: float) -> tuple[float, float, float, float]:
    radians = math.radians(degrees)
    return (math.cos(radians / 2.0), 0.0, math.sin(radians / 2.0), 0.0)


def create_v_groove_scene() -> PhysicsSceneSpec:
    """Create a symmetric V groove from two separate static ramp bodies.

    Each ramp is a 1.2 x 1.0 x 0.08 m box. The top planes meet near the
    world origin and are tilted by +/-35 degrees around world Y. Collision
    masks allow sphere-ramp contacts but block ramp-ramp contacts.
    """
    ramp_geometry = BoxGeometry((1.2, 1.0, 0.08))

    def ramp_asset(asset_id: str, body_id: str, *, angle_degrees: float, x: float, group: int):
        body = RigidBodySpec(
            body_id=body_id,
            name=body_id,
            body_type="static",
            transform=Transform(position=(x, 0.0, 0.161), rotation=qy(angle_degrees)),
            visuals=(),
            colliders=(
                ColliderSpec(
                    collider_id=f"{body_id}_collider",
                    geometry=ramp_geometry,
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
        scene_id="v_groove_contact",
        instances=(
            AssetInstanceSpec(
                "left_ramp",
                ramp_asset("left_ramp_asset", "left_ramp_body", angle_degrees=35.0, x=-0.3, group=2),
                fixed_base=True,
            ),
            AssetInstanceSpec(
                "right_ramp",
                ramp_asset("right_ramp_asset", "right_ramp_body", angle_degrees=-35.0, x=0.3, group=4),
                fixed_base=True,
            ),
            AssetInstanceSpec(
                "sphere",
                create_single_body_asset(asset_id="sphere_asset", body=sphere_body),
                Transform(position=(0.0, 0.0, 0.8)),
            ),
        ),
        gravity=(0.0, 0.0, -9.81),
        timestep=1.0 / 240.0,
    )


def create_offcenter_box_scene(angle_degrees: float) -> PhysicsSceneSpec:
    ground_asset = create_single_body_asset(asset_id="ground_asset", body=create_ground("ground_body"))
    box_asset = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.6, 0.2, 0.2), mass=1.0),
    )
    return create_scene(
        scene_id=f"offcenter_box_{angle_degrees:+.1f}",
        instances=(
            AssetInstanceSpec("ground", ground_asset, fixed_base=True),
            AssetInstanceSpec(
                "box",
                box_asset,
                Transform(position=(0.0, 0.0, 1.0), rotation=qy(angle_degrees)),
            ),
        ),
        gravity=(0.0, 0.0, -9.81),
        timestep=1.0 / 240.0,
    )


def active_wrenches_for_body(
    wrenches: tuple[ContactWrench, ...],
    runtime_body_id: str,
    *,
    threshold: float = ACTIVE_FORCE_THRESHOLD,
) -> tuple[ContactWrench, ...]:
    return tuple(
        wrench
        for wrench in wrenches
        if runtime_body_id in (wrench.contact.body_a, wrench.contact.body_b)
        and wrench.normal_force_magnitude > threshold
    )


def run_v_groove_to_rest(steps: int = 1500) -> tuple[MuJoCoBackend, RigidBodyState, tuple[ContactWrench, ...]]:
    backend = MuJoCoBackend()
    backend.load_scene(create_v_groove_scene())
    backend.reset()
    for _ in range(steps):
        backend.step()
    state = backend.get_body_state(SPHERE_BODY_ID)
    return backend, state, active_wrenches_for_body(backend.get_contact_wrenches(), SPHERE_BODY_ID)


def find_first_offcenter_impact(angle_degrees: float, *, max_steps: int = 600) -> ImpactSnapshot:
    backend = MuJoCoBackend()
    backend.load_scene(create_offcenter_box_scene(angle_degrees))
    previous_result = backend.reset()
    try:
        for _ in range(max_steps):
            current_result = backend.step()
            active = active_wrenches_for_body(backend.get_contact_wrenches(), BOX_BODY_ID)
            if active:
                post_result = backend.step()
                return ImpactSnapshot(
                    step_index=current_result.step_index,
                    time=current_result.time,
                    previous_state=previous_result.get_body_state(BOX_BODY_ID),
                    impact_state=current_result.get_body_state(BOX_BODY_ID),
                    post_impact_state=post_result.get_body_state(BOX_BODY_ID),
                    active_wrenches=active,
                )
            previous_result = current_result
    finally:
        backend.close()
    raise AssertionError(f"no active off-center impact found; angle_degrees={angle_degrees!r}")
