"""Physical asset data structures and parametric builders."""

from physical_simulation.assets.builders import (
    create_box,
    create_capsule,
    create_cylinder,
    create_ground,
    create_sphere,
)
from physical_simulation.assets.collider import ColliderSpec
from physical_simulation.assets.geometry import (
    BoxGeometry,
    CapsuleGeometry,
    CylinderGeometry,
    GeometrySpec,
    SphereGeometry,
    geometry_from_dict,
)
from physical_simulation.assets.mass_properties import MassProperties
from physical_simulation.assets.material import DEFAULT_MATERIAL, PhysicsMaterialSpec
from physical_simulation.assets.rigid_body import BodyType, RigidBodySpec
from physical_simulation.assets.transform import Transform
from physical_simulation.assets.visual import VisualSpec

__all__ = [
    "Transform",
    "BoxGeometry",
    "SphereGeometry",
    "CylinderGeometry",
    "CapsuleGeometry",
    "GeometrySpec",
    "geometry_from_dict",
    "PhysicsMaterialSpec",
    "DEFAULT_MATERIAL",
    "MassProperties",
    "VisualSpec",
    "ColliderSpec",
    "BodyType",
    "RigidBodySpec",
    "create_box",
    "create_sphere",
    "create_cylinder",
    "create_capsule",
    "create_ground",
]
