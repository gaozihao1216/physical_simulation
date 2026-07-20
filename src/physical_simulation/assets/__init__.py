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
from physical_simulation.assets.physics_asset import (
    CURRENT_PHYSICS_ASSET_SCHEMA_VERSION,
    PhysicsAssetSpec,
    create_single_body_asset,
)
from physical_simulation.assets.rigid_body import BodyType, RigidBodySpec
from physical_simulation.assets.scale_baking import (
    bake_scale_into_geometry,
    bake_transform_scale,
    is_unit_scale,
)
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
    "PhysicsAssetSpec",
    "CURRENT_PHYSICS_ASSET_SCHEMA_VERSION",
    "is_unit_scale",
    "bake_scale_into_geometry",
    "bake_transform_scale",
    "create_box",
    "create_sphere",
    "create_cylinder",
    "create_capsule",
    "create_ground",
    "create_single_body_asset",
]
