"""Compile PhysicsSceneSpec into MJCF XML without importing MuJoCo."""

from __future__ import annotations

import hashlib
import re
from xml.etree import ElementTree as ET

from physical_simulation.assets import (
    BoxGeometry,
    CapsuleGeometry,
    ColliderSpec,
    CylinderGeometry,
    GeometrySpec,
    PhysicsMaterialSpec,
    RigidBodySpec,
    SphereGeometry,
    Transform,
    VisualSpec,
)
from physical_simulation.compilers.errors import (
    MuJoCoCompilationError,
    UnsupportedAssetStructureError,
    UnsupportedPhysicsFeatureError,
)
from physical_simulation.compilers.mujoco_types import MuJoCoCompilationResult
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, PhysicsSceneSpec
from physical_simulation.utils import format_float, format_vector, indent_xml
from physical_simulation.validation.asset_validator import validate_physics_scene
from physical_simulation.validation.errors import PhysicsValidationError

MUJOCO_ALL_COLLISION_BITS = (1 << 31) - 1
MUJOCO_TORSIONAL_FRICTION = 0.005
MUJOCO_ROLLING_FRICTION = 0.0001


def make_mujoco_name(prefix: str, raw_id: str) -> str:
    """Create a stable MuJoCo-safe name using a short SHA-256 suffix."""
    safe_prefix = re.sub(r"[^A-Za-z0-9_]+", "_", prefix).strip("_") or "name"
    safe_raw = re.sub(r"[^A-Za-z0-9_]+", "_", raw_id).strip("_") or "id"
    digest = hashlib.sha256(f"{prefix}\0{raw_id}".encode("utf-8")).hexdigest()[:8]
    return f"{safe_prefix}_{safe_raw}_{digest}"


def geometry_to_mujoco(geometry: GeometrySpec) -> tuple[str, tuple[float, ...]]:
    """Map Physics IR primitive geometry to MuJoCo geom type and size."""
    if isinstance(geometry, BoxGeometry):
        x, y, z = geometry.size
        return "box", (x / 2.0, y / 2.0, z / 2.0)
    if isinstance(geometry, SphereGeometry):
        return "sphere", (geometry.radius,)
    if isinstance(geometry, CylinderGeometry):
        return "cylinder", (geometry.radius, geometry.height / 2.0)
    if isinstance(geometry, CapsuleGeometry):
        return "capsule", (geometry.radius, geometry.length / 2.0)
    raise UnsupportedPhysicsFeatureError(
        f"unsupported geometry for MuJoCo compilation; geometry={geometry!r}"
    )


def _mujoco_quat(rotation: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Return MuJoCo quaternion ordering. Project and MJCF both use w x y z."""
    return rotation


def _collision_mask_to_conaffinity(mask: int, *, scene_id: str, collider_id: str) -> int:
    if mask == -1:
        return MUJOCO_ALL_COLLISION_BITS
    if not isinstance(mask, int) or isinstance(mask, bool) or mask < 0:
        raise MuJoCoCompilationError(
            f"collision_mask must be -1 or a non-negative int within MuJoCo bit range; "
            f"scene_id={scene_id!r}, collider_id={collider_id!r}, actual value={mask!r}"
        )
    if mask > MUJOCO_ALL_COLLISION_BITS:
        raise MuJoCoCompilationError(
            f"collision_mask exceeds MuJoCo supported bits; scene_id={scene_id!r}, "
            f"collider_id={collider_id!r}, actual value={mask!r}, maximum={MUJOCO_ALL_COLLISION_BITS!r}"
        )
    return mask


def _collision_group_to_contype(group: int, *, scene_id: str, collider_id: str) -> int:
    if not isinstance(group, int) or isinstance(group, bool) or group < 0:
        raise MuJoCoCompilationError(
            f"collision_group must be a non-negative int bit mask; "
            f"scene_id={scene_id!r}, collider_id={collider_id!r}, actual value={group!r}"
        )
    if group > MUJOCO_ALL_COLLISION_BITS:
        raise MuJoCoCompilationError(
            f"collision_group exceeds MuJoCo supported bits; scene_id={scene_id!r}, "
            f"collider_id={collider_id!r}, actual value={group!r}, maximum={MUJOCO_ALL_COLLISION_BITS!r}"
        )
    return group


class MuJoCoCompiler:
    """Compile backend-independent scene specs into deterministic MJCF XML."""

    def compile(self, scene: PhysicsSceneSpec) -> MuJoCoCompilationResult:
        """Compile a PhysicsSceneSpec into MJCF and stable name mappings."""
        try:
            validate_physics_scene(scene)
        except PhysicsValidationError as exc:
            raise MuJoCoCompilationError(
                f"scene validation failed before MuJoCo compilation; scene={scene!r}; error={exc}"
            ) from exc

        root = ET.Element("mujoco", {"model": make_mujoco_name("model", scene.scene_id)})
        ET.SubElement(root, "compiler", {"angle": "radian"})
        ET.SubElement(
            root,
            "option",
            {
                "timestep": format_float(scene.timestep),
                "gravity": format_vector(scene.gravity),
            },
        )
        worldbody = ET.SubElement(root, "worldbody")

        body_mapping: list[tuple[str, str]] = []
        geom_mapping: list[tuple[str, str]] = []

        for instance in scene.instances:
            self._compile_instance(
                scene=scene,
                instance=instance,
                worldbody=worldbody,
                body_mapping=body_mapping,
                geom_mapping=geom_mapping,
            )

        indent_xml(root)
        mjcf = ET.tostring(root, encoding="unicode", short_empty_elements=True)
        return MuJoCoCompilationResult(
            scene_id=scene.scene_id,
            mjcf=mjcf,
            runtime_body_to_mujoco_name=tuple(body_mapping),
            mujoco_geom_to_runtime_body=tuple(geom_mapping),
        )

    def _compile_instance(
        self,
        *,
        scene: PhysicsSceneSpec,
        instance: AssetInstanceSpec,
        worldbody: ET.Element,
        body_mapping: list[tuple[str, str]],
        geom_mapping: list[tuple[str, str]],
    ) -> None:
        asset = instance.asset
        if len(asset.bodies) != 1:
            raise UnsupportedAssetStructureError(
                "MuJoCo Phase 2 supports exactly one rigid body per asset instance. "
                "Articulated multi-body assets require JointSpec support. "
                f"scene_id={scene.scene_id!r}, instance_id={instance.instance_id!r}, "
                f"asset_id={asset.asset_id!r}, body_count={len(asset.bodies)!r}"
            )
        body = asset.bodies[0]
        runtime_body_id = make_runtime_body_id(instance.instance_id, body.body_id)
        body_name = make_mujoco_name("body", runtime_body_id)
        world_transform = instance.transform.compose(body.transform)
        body_element = ET.SubElement(
            worldbody,
            "body",
            {
                "name": body_name,
                "pos": format_vector(world_transform.position),
                "quat": format_vector(_mujoco_quat(world_transform.rotation)),
            },
        )
        body_mapping.append((runtime_body_id, body_name))

        if body.body_type == "dynamic" and not instance.fixed_base:
            ET.SubElement(body_element, "freejoint")
        if body.body_type == "dynamic":
            self._compile_inertial(scene=scene, instance=instance, body=body, body_element=body_element)

        material_by_id = {material.material_id: material for material in asset.materials}
        for visual in body.visuals:
            self._compile_visual(
                instance=instance,
                body=body,
                visual=visual,
                body_element=body_element,
                runtime_body_id=runtime_body_id,
            )
        for collider in body.colliders:
            self._compile_collider(
                scene=scene,
                instance=instance,
                body=body,
                collider=collider,
                material_by_id=material_by_id,
                body_element=body_element,
                runtime_body_id=runtime_body_id,
                geom_mapping=geom_mapping,
            )

    def _compile_inertial(
        self,
        *,
        scene: PhysicsSceneSpec,
        instance: AssetInstanceSpec,
        body: RigidBodySpec,
        body_element: ET.Element,
    ) -> None:
        mass_properties = body.mass_properties
        if mass_properties is None:
            raise MuJoCoCompilationError(
                f"dynamic body requires MassProperties for inertial compilation; "
                f"scene_id={scene.scene_id!r}, instance_id={instance.instance_id!r}, body_id={body.body_id!r}"
            )
        ET.SubElement(
            body_element,
            "inertial",
            {
                "pos": format_vector(mass_properties.center_of_mass),
                "mass": format_float(mass_properties.mass),
                "diaginertia": format_vector(mass_properties.inertia_diagonal),
            },
        )

    def _compile_visual(
        self,
        *,
        instance: AssetInstanceSpec,
        body: RigidBodySpec,
        visual: VisualSpec,
        body_element: ET.Element,
        runtime_body_id: str,
    ) -> None:
        if not visual.visible:
            return
        geom_type, geom_size = geometry_to_mujoco(visual.geometry)
        ET.SubElement(
            body_element,
            "geom",
            {
                "name": make_mujoco_name("visual", f"{runtime_body_id}/{visual.visual_id}"),
                "type": geom_type,
                "size": format_vector(geom_size),
                "pos": format_vector(visual.local_transform.position),
                "quat": format_vector(_mujoco_quat(visual.local_transform.rotation)),
                "contype": "0",
                "conaffinity": "0",
                "rgba": "0.7 0.7 0.7 1",
            },
        )

    def _compile_collider(
        self,
        *,
        scene: PhysicsSceneSpec,
        instance: AssetInstanceSpec,
        body: RigidBodySpec,
        collider: ColliderSpec,
        material_by_id: dict[str, PhysicsMaterialSpec],
        body_element: ET.Element,
        runtime_body_id: str,
        geom_mapping: list[tuple[str, str]],
    ) -> None:
        if not collider.enabled:
            return
        material = material_by_id.get(collider.material_id)
        if material is None:
            raise MuJoCoCompilationError(
                f"collider material_id is missing from asset materials; scene_id={scene.scene_id!r}, "
                f"instance_id={instance.instance_id!r}, asset_id={instance.asset.asset_id!r}, "
                f"body_id={body.body_id!r}, collider_id={collider.collider_id!r}, "
                f"material_id={collider.material_id!r}"
            )
        geom_type, geom_size = geometry_to_mujoco(collider.geometry)
        geom_name = make_mujoco_name("collision", f"{runtime_body_id}/{collider.collider_id}")
        # MuJoCo friction is an approximation here: sliding uses dynamic_friction;
        # static_friction and restitution are retained in IR but not directly mapped.
        ET.SubElement(
            body_element,
            "geom",
            {
                "name": geom_name,
                "type": geom_type,
                "size": format_vector(geom_size),
                "pos": format_vector(collider.local_transform.position),
                "quat": format_vector(_mujoco_quat(collider.local_transform.rotation)),
                "contype": str(
                    _collision_group_to_contype(
                        collider.collision_group,
                        scene_id=scene.scene_id,
                        collider_id=collider.collider_id,
                    )
                ),
                "conaffinity": str(
                    _collision_mask_to_conaffinity(
                        collider.collision_mask,
                        scene_id=scene.scene_id,
                        collider_id=collider.collider_id,
                    )
                ),
                "friction": format_vector(
                    (
                        material.dynamic_friction,
                        MUJOCO_TORSIONAL_FRICTION,
                        MUJOCO_ROLLING_FRICTION,
                    )
                ),
            },
        )
        geom_mapping.append((geom_name, runtime_body_id))
