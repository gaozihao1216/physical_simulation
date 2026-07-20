"""Parametric builders for basic rigid body specifications."""

from __future__ import annotations

from typing import Optional

from physical_simulation.assets.collider import ColliderSpec
from physical_simulation.assets.geometry import (
    BoxGeometry,
    CapsuleGeometry,
    CylinderGeometry,
    GeometrySpec,
    SphereGeometry,
)
from physical_simulation.assets.mass_properties import MassProperties
from physical_simulation.assets.material import DEFAULT_MATERIAL, PhysicsMaterialSpec
from physical_simulation.assets.rigid_body import BodyType, RigidBodySpec
from physical_simulation.assets.transform import Transform
from physical_simulation.assets.visual import VisualSpec
from physical_simulation.dynamics.inertia import compute_inertia, compute_mass_from_density
from physical_simulation.validation.errors import InvalidRigidBodyError


def _resolve_material(material: Optional[PhysicsMaterialSpec]) -> PhysicsMaterialSpec:
    return DEFAULT_MATERIAL if material is None else material


def _resolve_mass(
    *,
    geometry: GeometrySpec,
    body_type: BodyType,
    mass: Optional[float],
    density: Optional[float],
    material: PhysicsMaterialSpec,
) -> Optional[MassProperties]:
    if body_type != "dynamic":
        return None
    if mass is not None and density is not None:
        raise InvalidRigidBodyError(
            f"mass and density are mutually exclusive; actual mass={mass!r}, density={density!r}"
        )
    if mass is None:
        selected_density = density if density is not None else material.density
        if selected_density is None:
            raise InvalidRigidBodyError(
                "dynamic body requires mass, density, or material.density; actual values are all None"
            )
        mass = compute_mass_from_density(geometry, selected_density)
    inertia = compute_inertia(geometry, mass)
    return MassProperties(
        mass=mass,
        center_of_mass=(0.0, 0.0, 0.0),
        inertia_diagonal=inertia,
    )


def _build_body(
    *,
    body_id: str,
    geometry: GeometrySpec,
    name: Optional[str],
    body_type: BodyType,
    transform: Optional[Transform],
    mass: Optional[float],
    density: Optional[float],
    material: Optional[PhysicsMaterialSpec],
    create_visual: bool,
    create_collider: bool,
) -> RigidBodySpec:
    resolved_material = _resolve_material(material)
    resolved_transform = Transform.identity() if transform is None else transform
    visuals = (
        (VisualSpec(visual_id=f"{body_id}_visual", geometry=geometry),)
        if create_visual
        else ()
    )
    colliders = (
        (
            ColliderSpec(
                collider_id=f"{body_id}_collider",
                geometry=geometry,
                material_id=resolved_material.material_id,
            ),
        )
        if create_collider
        else ()
    )
    return RigidBodySpec(
        body_id=body_id,
        name=name or body_id,
        body_type=body_type,
        transform=resolved_transform,
        visuals=visuals,
        colliders=colliders,
        mass_properties=_resolve_mass(
            geometry=geometry,
            body_type=body_type,
            mass=mass,
            density=density,
            material=resolved_material,
        ),
    )


def create_box(
    body_id: str,
    size: tuple[float, float, float],
    *,
    name: Optional[str] = None,
    body_type: BodyType = "dynamic",
    transform: Optional[Transform] = None,
    mass: Optional[float] = None,
    density: Optional[float] = None,
    material: Optional[PhysicsMaterialSpec] = None,
    create_visual: bool = True,
    create_collider: bool = True,
) -> RigidBodySpec:
    """Create a box rigid body specification."""
    return _build_body(
        body_id=body_id,
        geometry=BoxGeometry(size=size),
        name=name,
        body_type=body_type,
        transform=transform,
        mass=mass,
        density=density,
        material=material,
        create_visual=create_visual,
        create_collider=create_collider,
    )


def create_sphere(
    body_id: str,
    radius: float,
    *,
    name: Optional[str] = None,
    body_type: BodyType = "dynamic",
    transform: Optional[Transform] = None,
    mass: Optional[float] = None,
    density: Optional[float] = None,
    material: Optional[PhysicsMaterialSpec] = None,
    create_visual: bool = True,
    create_collider: bool = True,
) -> RigidBodySpec:
    """Create a sphere rigid body specification."""
    return _build_body(
        body_id=body_id,
        geometry=SphereGeometry(radius=radius),
        name=name,
        body_type=body_type,
        transform=transform,
        mass=mass,
        density=density,
        material=material,
        create_visual=create_visual,
        create_collider=create_collider,
    )


def create_cylinder(
    body_id: str,
    radius: float,
    height: float,
    *,
    name: Optional[str] = None,
    body_type: BodyType = "dynamic",
    transform: Optional[Transform] = None,
    mass: Optional[float] = None,
    density: Optional[float] = None,
    material: Optional[PhysicsMaterialSpec] = None,
    create_visual: bool = True,
    create_collider: bool = True,
) -> RigidBodySpec:
    """Create a cylinder rigid body specification."""
    return _build_body(
        body_id=body_id,
        geometry=CylinderGeometry(radius=radius, height=height),
        name=name,
        body_type=body_type,
        transform=transform,
        mass=mass,
        density=density,
        material=material,
        create_visual=create_visual,
        create_collider=create_collider,
    )


def create_capsule(
    body_id: str,
    radius: float,
    length: float,
    *,
    name: Optional[str] = None,
    body_type: BodyType = "dynamic",
    transform: Optional[Transform] = None,
    mass: Optional[float] = None,
    density: Optional[float] = None,
    material: Optional[PhysicsMaterialSpec] = None,
    create_visual: bool = True,
    create_collider: bool = True,
) -> RigidBodySpec:
    """Create a capsule rigid body specification."""
    return _build_body(
        body_id=body_id,
        geometry=CapsuleGeometry(radius=radius, length=length),
        name=name,
        body_type=body_type,
        transform=transform,
        mass=mass,
        density=density,
        material=material,
        create_visual=create_visual,
        create_collider=create_collider,
    )


def create_ground(
    body_id: str = "ground",
    *,
    name: Optional[str] = None,
    size: tuple[float, float, float] = (20.0, 20.0, 0.1),
    transform: Optional[Transform] = None,
    material: Optional[PhysicsMaterialSpec] = None,
) -> RigidBodySpec:
    """Create a static finite ground box.

    This is a temporary finite ground representation, not an infinite plane.
    """
    ground_transform = transform or Transform(position=(0.0, 0.0, -size[2] / 2.0))
    return create_box(
        body_id=body_id,
        size=size,
        name=name or body_id,
        body_type="static",
        transform=ground_transform,
        material=material,
    )
