"""Compile PhysicsSceneSpec into MJCF XML without importing MuJoCo."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
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
from physical_simulation.collision.convex_mesh import geometry_to_convex_mesh, supports_mujoco_mesh_fallback
from physical_simulation.compilers.mujoco_types import MuJoCoCompilationResult
from physical_simulation.mujoco.contact_params import MuJoCoContactSolverParams
from physical_simulation.runtime import make_runtime_body_id
from physical_simulation.scene import AssetInstanceSpec, PhysicsSceneSpec
from physical_simulation.utils import format_float, format_vector, indent_xml
from physical_simulation.validation.asset_validator import validate_physics_scene
from physical_simulation.validation.errors import PhysicsValidationError

MUJOCO_ALL_COLLISION_BITS = (1 << 31) - 1
MUJOCO_TORSIONAL_FRICTION = 0.005
MUJOCO_ROLLING_FRICTION = 0.0001
MUJOCO_EXPLICIT_PAIR_CONDIM = 3
MUJOCO_EXPLICIT_PAIR_MARGIN = 0.0
MUJOCO_EXPLICIT_PAIR_GAP = 0.0


@dataclass(frozen=True)
class _ContactPairCandidate:
    geom_name: str
    runtime_body_id: str
    contype: int
    conaffinity: int
    has_dof: bool
    material: PhysicsMaterialSpec
    contact_params: MuJoCoContactSolverParams | None


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
        asset_element = ET.Element("asset")
        mesh_assets: dict[str, str] = {}

        body_mapping: list[tuple[str, str]] = []
        geom_mapping: list[tuple[str, str]] = []
        contact_pair_candidates: list[_ContactPairCandidate] = []

        for instance in scene.instances:
            self._compile_instance(
                scene=scene,
                instance=instance,
                worldbody=worldbody,
                asset_element=asset_element,
                mesh_assets=mesh_assets,
                body_mapping=body_mapping,
                geom_mapping=geom_mapping,
                contact_pair_candidates=contact_pair_candidates,
            )

        if len(asset_element):
            root.insert(2, asset_element)
        self._compile_contact_pairs(root, contact_pair_candidates)
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
        asset_element: ET.Element,
        mesh_assets: dict[str, str],
        body_mapping: list[tuple[str, str]],
        geom_mapping: list[tuple[str, str]],
        contact_pair_candidates: list[_ContactPairCandidate],
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
                asset_element=asset_element,
                mesh_assets=mesh_assets,
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
                asset_element=asset_element,
                mesh_assets=mesh_assets,
                runtime_body_id=runtime_body_id,
                geom_mapping=geom_mapping,
                contact_pair_candidates=contact_pair_candidates,
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
        attributes = {
            "pos": format_vector(mass_properties.center_of_mass),
            "mass": format_float(mass_properties.mass),
            "diaginertia": format_vector(mass_properties.inertia_diagonal),
        }
        if mass_properties.has_non_identity_principal_axes:
            attributes["quat"] = format_vector(_mujoco_quat(mass_properties.inertial_frame_quaternion))
        ET.SubElement(body_element, "inertial", attributes)

    def _compile_visual(
        self,
        *,
        instance: AssetInstanceSpec,
        body: RigidBodySpec,
        visual: VisualSpec,
        body_element: ET.Element,
        asset_element: ET.Element,
        mesh_assets: dict[str, str],
        runtime_body_id: str,
    ) -> None:
        if not visual.visible:
            return
        geom_attributes = {
                "name": make_mujoco_name("visual", f"{runtime_body_id}/{visual.visual_id}"),
                "pos": format_vector(visual.local_transform.position),
                "quat": format_vector(_mujoco_quat(visual.local_transform.rotation)),
                "contype": "0",
                "conaffinity": "0",
                "rgba": "0.7 0.7 0.7 1",
        }
        geom_attributes.update(
            self._geometry_geom_attributes(
                visual.geometry,
                asset_element=asset_element,
                mesh_assets=mesh_assets,
            )
        )
        ET.SubElement(body_element, "geom", geom_attributes)

    def _compile_collider(
        self,
        *,
        scene: PhysicsSceneSpec,
        instance: AssetInstanceSpec,
        body: RigidBodySpec,
        collider: ColliderSpec,
        material_by_id: dict[str, PhysicsMaterialSpec],
        body_element: ET.Element,
        asset_element: ET.Element,
        mesh_assets: dict[str, str],
        runtime_body_id: str,
        geom_mapping: list[tuple[str, str]],
        contact_pair_candidates: list[_ContactPairCandidate],
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
        geom_name = make_mujoco_name("collision", f"{runtime_body_id}/{collider.collider_id}")
        contype = _collision_group_to_contype(
            collider.collision_group,
            scene_id=scene.scene_id,
            collider_id=collider.collider_id,
        )
        conaffinity = _collision_mask_to_conaffinity(
            collider.collision_mask,
            scene_id=scene.scene_id,
            collider_id=collider.collider_id,
        )
        # MuJoCo friction is an approximation here: sliding uses dynamic_friction;
        # static_friction and restitution are retained in IR but not directly mapped.
        geom_attributes = {
                "name": geom_name,
                "pos": format_vector(collider.local_transform.position),
                "quat": format_vector(_mujoco_quat(collider.local_transform.rotation)),
                "contype": str(contype),
                "conaffinity": str(conaffinity),
                "friction": format_vector(
                    (
                        material.dynamic_friction,
                        MUJOCO_TORSIONAL_FRICTION,
                        MUJOCO_ROLLING_FRICTION,
                    )
                ),
        }
        geom_attributes.update(self._mujoco_contact_param_attributes(collider.mujoco_contact_params))
        geom_attributes.update(
            self._geometry_geom_attributes(
                collider.geometry,
                asset_element=asset_element,
                mesh_assets=mesh_assets,
            )
        )
        ET.SubElement(body_element, "geom", geom_attributes)
        geom_mapping.append((geom_name, runtime_body_id))
        contact_pair_candidates.append(
            _ContactPairCandidate(
                geom_name=geom_name,
                runtime_body_id=runtime_body_id,
                contype=contype,
                conaffinity=conaffinity,
                has_dof=self._body_has_dof(body=body, instance=instance),
                material=material,
                contact_params=collider.mujoco_contact_params,
            )
        )

    def _mujoco_contact_param_attributes(
        self,
        params: MuJoCoContactSolverParams | None,
    ) -> dict[str, str]:
        if params is None:
            return {}
        return {
            "solref": format_vector(params.solref),
            "solimp": format_vector(params.solimp),
            "margin": format_float(params.margin),
            "gap": format_float(params.gap),
            "priority": str(params.priority),
            "solmix": format_float(params.solmix),
        }

    def _geometry_geom_attributes(
        self,
        geometry: GeometrySpec,
        *,
        asset_element: ET.Element,
        mesh_assets: dict[str, str],
    ) -> dict[str, str]:
        try:
            geom_type, geom_size = geometry_to_mujoco(geometry)
            return {"type": geom_type, "size": format_vector(geom_size)}
        except UnsupportedPhysicsFeatureError:
            if not supports_mujoco_mesh_fallback(geometry):
                raise
        mesh_name = self._ensure_mesh_asset(
            geometry,
            asset_element=asset_element,
            mesh_assets=mesh_assets,
        )
        return {"type": "mesh", "mesh": mesh_name}

    def _ensure_mesh_asset(
        self,
        geometry: GeometrySpec,
        *,
        asset_element: ET.Element,
        mesh_assets: dict[str, str],
    ) -> str:
        key = json.dumps(geometry.to_dict(), sort_keys=True, separators=(",", ":"))
        if key in mesh_assets:
            return mesh_assets[key]
        mesh = geometry_to_convex_mesh(geometry)
        mesh_name = make_mujoco_name("mesh", key)
        ET.SubElement(
            asset_element,
            "mesh",
            {
                "name": mesh_name,
                "vertex": mesh.vertex_attribute(),
                "face": mesh.face_attribute(),
            },
        )
        mesh_assets[key] = mesh_name
        return mesh_name

    def _compile_contact_pairs(
        self,
        root: ET.Element,
        contact_pair_candidates: list[_ContactPairCandidate],
    ) -> None:
        pair_elements: dict[tuple[str, str], dict[str, str]] = {}
        for index, first in enumerate(contact_pair_candidates):
            for second in contact_pair_candidates[index + 1:]:
                if first.runtime_body_id == second.runtime_body_id:
                    continue
                if not self._explicit_contact_pair_supported(first, second):
                    continue
                if not self._collision_pair_enabled(
                    first.contype,
                    first.conaffinity,
                    second.contype,
                    second.conaffinity,
                ):
                    continue
                key = self._canonical_pair_key(first.geom_name, second.geom_name)
                pair_elements[key] = self._explicit_pair_attributes(first, second, key)
        if not pair_elements:
            return
        contact_element = ET.SubElement(root, "contact")
        for key in sorted(pair_elements):
            ET.SubElement(contact_element, "pair", pair_elements[key])

    def _collision_pair_enabled(
        self,
        first_contype: int,
        first_conaffinity: int,
        second_contype: int,
        second_conaffinity: int,
    ) -> bool:
        return bool(
            (first_contype & second_conaffinity)
            or (second_contype & first_conaffinity)
        )

    def _body_has_dof(self, *, body: RigidBodySpec, instance: AssetInstanceSpec) -> bool:
        return body.body_type == "dynamic" and not instance.fixed_base

    def _explicit_contact_pair_supported(
        self,
        first: _ContactPairCandidate,
        second: _ContactPairCandidate,
    ) -> bool:
        # Explicit pairs are only for no-DOF body pairs that MuJoCo's dynamic
        # broadphase does not reliably expose as active contacts. If either
        # side is a normal free dynamic body, contype/conaffinity should handle
        # collision without an explicit pair.
        return not first.has_dof and not second.has_dof

    def _canonical_pair_key(self, first_geom: str, second_geom: str) -> tuple[str, str]:
        return (first_geom, second_geom) if first_geom <= second_geom else (second_geom, first_geom)

    def _explicit_pair_attributes(
        self,
        first: _ContactPairCandidate,
        second: _ContactPairCandidate,
        key: tuple[str, str],
    ) -> dict[str, str]:
        attributes = {
            "geom1": key[0],
            "geom2": key[1],
            "condim": str(MUJOCO_EXPLICIT_PAIR_CONDIM),
            "friction": format_vector(self._mix_pair_friction(first.material, second.material)),
            "margin": format_float(self._explicit_pair_margin(first.contact_params, second.contact_params)),
            "gap": format_float(self._explicit_pair_gap(first.contact_params, second.contact_params)),
        }
        solver_params = self._resolve_explicit_pair_solver_params(first.contact_params, second.contact_params)
        if solver_params is not None:
            attributes["solref"] = format_vector(solver_params.solref)
            attributes["solimp"] = format_vector(solver_params.solimp)
        return attributes

    def _explicit_pair_margin(
        self,
        first: MuJoCoContactSolverParams | None,
        second: MuJoCoContactSolverParams | None,
    ) -> float:
        return (first.margin if first is not None else MUJOCO_EXPLICIT_PAIR_MARGIN) + (
            second.margin if second is not None else MUJOCO_EXPLICIT_PAIR_MARGIN
        )

    def _explicit_pair_gap(
        self,
        first: MuJoCoContactSolverParams | None,
        second: MuJoCoContactSolverParams | None,
    ) -> float:
        return (first.gap if first is not None else MUJOCO_EXPLICIT_PAIR_GAP) + (
            second.gap if second is not None else MUJOCO_EXPLICIT_PAIR_GAP
        )

    def _resolve_explicit_pair_solver_params(
        self,
        first: MuJoCoContactSolverParams | None,
        second: MuJoCoContactSolverParams | None,
    ) -> MuJoCoContactSolverParams | None:
        if first is None and second is None:
            return None
        if first is None:
            return second
        if second is None:
            return first
        if first.priority > second.priority:
            return first
        if second.priority > first.priority:
            return second
        return MuJoCoContactSolverParams(
            solref=self._mix_equal_priority_solref(first, second),
            solimp=self._weighted_average(first.solimp, first.solmix, second.solimp, second.solmix),
            margin=self._explicit_pair_margin(first, second),
            gap=self._explicit_pair_gap(first, second),
            priority=first.priority,
            solmix=max(first.solmix, second.solmix),
        )

    def _mix_equal_priority_solref(
        self,
        first: MuJoCoContactSolverParams,
        second: MuJoCoContactSolverParams,
    ) -> tuple[float, float]:
        if first.solref[0] <= 0.0 or second.solref[0] <= 0.0:
            return tuple(min(first.solref[index], second.solref[index]) for index in range(2))  # type: ignore[return-value]
        return self._weighted_average(first.solref, first.solmix, second.solref, second.solmix)

    def _weighted_average(
        self,
        first_values: tuple[float, ...],
        first_weight: float,
        second_values: tuple[float, ...],
        second_weight: float,
    ) -> tuple[float, ...]:
        weight_sum = first_weight + second_weight
        if weight_sum == 0.0:
            return tuple((first + second) * 0.5 for first, second in zip(first_values, second_values))
        return tuple(
            (first * first_weight + second * second_weight) / weight_sum
            for first, second in zip(first_values, second_values)
        )

    def _mix_pair_friction(
        self,
        first_material: PhysicsMaterialSpec,
        second_material: PhysicsMaterialSpec,
    ) -> tuple[float, float, float]:
        sliding = math.sqrt(first_material.dynamic_friction * second_material.dynamic_friction)
        return (
            sliding,
            MUJOCO_TORSIONAL_FRICTION,
            MUJOCO_ROLLING_FRICTION,
        )
