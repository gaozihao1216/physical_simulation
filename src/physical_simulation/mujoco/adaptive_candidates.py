"""Build adaptive prediction candidates from compiled Physics IR scenes."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from physical_simulation.assets import BoxGeometry, GeometrySpec, SphereGeometry
from physical_simulation.backends.mujoco_backend import MuJoCoBackend
from physical_simulation.compilers.mujoco_compiler import collision_pair_enabled
from physical_simulation.compilers.mujoco_types import CompiledColliderMetadata
from physical_simulation.mujoco.adaptive import (
    AdaptiveCollisionCandidate,
    AdaptiveMuJoCoRunner,
    AdaptiveSubstepConfig,
    SpherePlaneAdaptiveCandidate,
    SphereSphereAdaptiveCandidate,
)
from physical_simulation.mujoco.collision_prediction import AnalyticPlane
from physical_simulation.mujoco.contact_params import MuJoCoContactSolverParams, resolve_mujoco_contact_solver_params
from physical_simulation.scene import PhysicsSceneSpec
from physical_simulation.validation.asset_validator import _finite_float
from physical_simulation.validation.errors import PhysicsValidationError

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class AdaptiveCandidateBuildConfig:
    """Configuration for automatic adaptive candidate construction."""

    include_sphere_plane: bool = True
    include_sphere_sphere: bool = True
    require_at_least_one_dynamic_body: bool = True
    exclude_same_runtime_body: bool = True
    respect_collision_masks: bool = True
    allow_box_top_surface_as_plane: bool = True
    maximum_plane_tilt_degrees: float = 5.0
    minimum_plane_half_extent: float = 0.25
    minimum_plane_extent_to_sphere_radius_ratio: float = 3.0
    fail_on_unsupported_geometry: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "maximum_plane_tilt_degrees",
            "minimum_plane_half_extent",
            "minimum_plane_extent_to_sphere_radius_ratio",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_float(
                    getattr(self, field_name),
                    field_name=field_name,
                    minimum=0.0,
                    error_type=PhysicsValidationError,
                ),
            )


class AdaptiveCandidateDiagnosticStatus(Enum):
    """Status for one inspected collider pair."""

    GENERATED = "generated"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class AdaptiveCandidateDiagnostic:
    """Diagnostic for one candidate-builder collider pair decision."""

    collider_a_id: str
    collider_b_id: str
    runtime_body_a_id: str
    runtime_body_b_id: str
    geometry_a_type: str
    geometry_b_type: str
    status: AdaptiveCandidateDiagnosticStatus
    reason: str
    generated_candidate_id: str | None


@dataclass(frozen=True)
class AdaptiveCandidateBuildResult:
    """Result of automatic adaptive prediction candidate construction."""

    candidates: tuple[AdaptiveCollisionCandidate, ...]
    inspected_collider_count: int
    eligible_pair_count: int
    generated_candidate_count: int
    skipped_same_body_count: int
    skipped_mask_count: int
    skipped_static_static_count: int
    skipped_unsupported_geometry_count: int
    skipped_invalid_plane_count: int
    diagnostics: tuple[AdaptiveCandidateDiagnostic, ...]


def build_adaptive_prediction_candidates(
    *,
    scene: PhysicsSceneSpec,
    backend: MuJoCoBackend,
    config: AdaptiveCandidateBuildConfig | None = None,
) -> AdaptiveCandidateBuildResult:
    """Build supported adaptive prediction candidates from compiled scene metadata."""
    cfg = config or AdaptiveCandidateBuildConfig()
    if not isinstance(backend, MuJoCoBackend):
        raise PhysicsValidationError(f"backend must be MuJoCoBackend; actual value={backend!r}")
    if backend.scene is not scene:
        # Value equality is enough for dataclass scene copies; the important part
        # is that metadata comes from the same loaded scene semantics.
        if backend.scene != scene:
            raise PhysicsValidationError("backend must be loaded with the same scene passed to candidate builder")
    compilation = backend.compilation_result
    if compilation is None:
        raise PhysicsValidationError("backend must be loaded before building adaptive candidates")

    colliders = tuple(sorted(compilation.collider_metadata, key=_metadata_sort_key))
    candidates: dict[str, AdaptiveCollisionCandidate] = {}
    diagnostics: list[AdaptiveCandidateDiagnostic] = []
    skipped_same = 0
    skipped_mask = 0
    skipped_static = 0
    skipped_unsupported = 0
    skipped_invalid_plane = 0
    eligible = 0
    seen_pairs: set[tuple[str, str]] = set()

    for index, first in enumerate(colliders):
        for second in colliders[index + 1:]:
            pair_key = _canonical_pair_key(first, second)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            if cfg.exclude_same_runtime_body and first.runtime_body_id == second.runtime_body_id:
                skipped_same += 1
                diagnostics.append(_diagnostic(first, second, AdaptiveCandidateDiagnosticStatus.SKIPPED, "same runtime body", None))
                continue
            if cfg.require_at_least_one_dynamic_body and not (first.is_dynamic or second.is_dynamic):
                skipped_static += 1
                diagnostics.append(_diagnostic(first, second, AdaptiveCandidateDiagnosticStatus.SKIPPED, "both bodies have no free dynamic DOF", None))
                continue
            if cfg.respect_collision_masks and not collision_pair_enabled(
                first.collision_group,
                first.collision_mask,
                second.collision_group,
                second.collision_mask,
            ):
                skipped_mask += 1
                diagnostics.append(_diagnostic(first, second, AdaptiveCandidateDiagnosticStatus.SKIPPED, "collision group/mask disallow pair", None))
                continue

            eligible += 1
            candidate, reason = _build_pair_candidate(first, second, cfg)
            if candidate is None:
                if reason.startswith("invalid plane"):
                    skipped_invalid_plane += 1
                    status = AdaptiveCandidateDiagnosticStatus.SKIPPED
                else:
                    skipped_unsupported += 1
                    status = AdaptiveCandidateDiagnosticStatus.UNSUPPORTED
                    if cfg.fail_on_unsupported_geometry:
                        raise PhysicsValidationError(
                            f"unsupported adaptive candidate geometry pair; first={first.geometry!r}, second={second.geometry!r}, reason={reason}"
                        )
                diagnostics.append(_diagnostic(first, second, status, reason, None))
                continue
            candidates.setdefault(candidate.candidate_id, candidate)
            diagnostics.append(_diagnostic(first, second, AdaptiveCandidateDiagnosticStatus.GENERATED, reason, candidate.candidate_id))

    ordered = tuple(sorted(candidates.values(), key=lambda item: (_candidate_type_name(item), item.candidate_id)))
    return AdaptiveCandidateBuildResult(
        candidates=ordered,
        inspected_collider_count=len(colliders),
        eligible_pair_count=eligible,
        generated_candidate_count=len(ordered),
        skipped_same_body_count=skipped_same,
        skipped_mask_count=skipped_mask,
        skipped_static_static_count=skipped_static,
        skipped_unsupported_geometry_count=skipped_unsupported,
        skipped_invalid_plane_count=skipped_invalid_plane,
        diagnostics=tuple(sorted(diagnostics, key=lambda item: (item.collider_a_id, item.collider_b_id, item.status.value, item.reason))),
    )


def create_adaptive_runner_from_scene(
    backend: MuJoCoBackend,
    *,
    scene: PhysicsSceneSpec,
    runner_config: AdaptiveSubstepConfig,
    candidate_build_config: AdaptiveCandidateBuildConfig | None = None,
    manual_candidates: Sequence[AdaptiveCollisionCandidate] = (),
) -> tuple[AdaptiveMuJoCoRunner, AdaptiveCandidateBuildResult]:
    """Create an AdaptiveMuJoCoRunner using auto candidates plus optional manual candidates.

    Manual candidates win when candidate IDs collide.
    """
    result = build_adaptive_prediction_candidates(scene=scene, backend=backend, config=candidate_build_config)
    by_id = {candidate.candidate_id: candidate for candidate in result.candidates}
    for candidate in manual_candidates:
        by_id[candidate.candidate_id] = candidate
    candidates = tuple(sorted(by_id.values(), key=lambda item: (_candidate_type_name(item), item.candidate_id)))
    return AdaptiveMuJoCoRunner(backend, candidates=candidates, config=runner_config), result


def _build_pair_candidate(
    first: CompiledColliderMetadata,
    second: CompiledColliderMetadata,
    config: AdaptiveCandidateBuildConfig,
) -> tuple[AdaptiveCollisionCandidate | None, str]:
    ordered_first, ordered_second = _canonical_metadata_pair(first, second)
    if config.include_sphere_sphere and isinstance(ordered_first.geometry, SphereGeometry) and isinstance(ordered_second.geometry, SphereGeometry):
        return _sphere_sphere_candidate(ordered_first, ordered_second), "generated sphere-sphere candidate"
    if config.include_sphere_plane:
        candidate = _sphere_plane_candidate(first, second, config)
        if candidate is not None:
            return candidate, "generated sphere-plane candidate from static box top"
        reason = _sphere_plane_skip_reason(first, second, config)
        if reason is not None:
            return None, reason
    return None, "unsupported geometry pair for automatic adaptive prediction"


def _sphere_sphere_candidate(
    first: CompiledColliderMetadata,
    second: CompiledColliderMetadata,
) -> SphereSphereAdaptiveCandidate:
    candidate_id = _candidate_id("sphere_sphere", first, second)
    return SphereSphereAdaptiveCandidate(
        candidate_id=candidate_id,
        body_a_id=first.runtime_body_id,
        radius_a=first.geometry.radius,  # type: ignore[union-attr]
        body_b_id=second.runtime_body_id,
        radius_b=second.geometry.radius,  # type: ignore[union-attr]
        contact_params=resolve_mujoco_contact_solver_params(first.contact_params, second.contact_params),
    )


def _sphere_plane_candidate(
    first: CompiledColliderMetadata,
    second: CompiledColliderMetadata,
    config: AdaptiveCandidateBuildConfig,
) -> SpherePlaneAdaptiveCandidate | None:
    sphere, box = _sphere_box_pair(first, second)
    if sphere is None or box is None or not config.allow_box_top_surface_as_plane:
        return None
    if not sphere.is_dynamic or box.is_dynamic:
        return None
    valid, _reason, plane = _box_top_plane(box, sphere, config)
    if not valid or plane is None:
        return None
    return SpherePlaneAdaptiveCandidate(
        candidate_id=_candidate_id("sphere_plane", sphere, box),
        sphere_runtime_body_id=sphere.runtime_body_id,
        sphere_radius=sphere.geometry.radius,  # type: ignore[union-attr]
        plane=plane,
        contact_params=resolve_mujoco_contact_solver_params(sphere.contact_params, box.contact_params),
    )


def _sphere_plane_skip_reason(
    first: CompiledColliderMetadata,
    second: CompiledColliderMetadata,
    config: AdaptiveCandidateBuildConfig,
) -> str | None:
    sphere, box = _sphere_box_pair(first, second)
    if sphere is None or box is None:
        return None
    if not config.allow_box_top_surface_as_plane:
        return "unsupported sphere-box plane approximation disabled"
    if not sphere.is_dynamic or box.is_dynamic:
        return "unsupported sphere-plane candidate requires dynamic sphere and static box"
    valid, reason, _plane = _box_top_plane(box, sphere, config)
    if not valid:
        return f"invalid plane: {reason}"
    return None


def _sphere_box_pair(
    first: CompiledColliderMetadata,
    second: CompiledColliderMetadata,
) -> tuple[CompiledColliderMetadata | None, CompiledColliderMetadata | None]:
    if isinstance(first.geometry, SphereGeometry) and isinstance(second.geometry, BoxGeometry):
        return first, second
    if isinstance(second.geometry, SphereGeometry) and isinstance(first.geometry, BoxGeometry):
        return second, first
    return None, None


def _box_top_plane(
    box: CompiledColliderMetadata,
    sphere: CompiledColliderMetadata,
    config: AdaptiveCandidateBuildConfig,
) -> tuple[bool, str, AnalyticPlane | None]:
    geometry = box.geometry
    if not isinstance(geometry, BoxGeometry) or not isinstance(sphere.geometry, SphereGeometry):
        return False, "geometry is not static box plus sphere", None
    sx, sy, sz = geometry.size
    half_u = sx / 2.0
    half_v = sy / 2.0
    min_half = min(half_u, half_v)
    if min_half < config.minimum_plane_half_extent:
        return False, "box top half extent below minimum", None
    if min_half < config.minimum_plane_extent_to_sphere_radius_ratio * sphere.geometry.radius:
        return False, "box top extent too small relative to sphere radius", None
    normal = box.world_transform.rotate_vector((0.0, 0.0, 1.0))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, _dot(_normalize(normal), (0.0, 0.0, 1.0))))))
    if tilt > config.maximum_plane_tilt_degrees:
        return False, "box top normal exceeds maximum tilt", None
    axis_u = _normalize(box.world_transform.rotate_vector((1.0, 0.0, 0.0)))
    axis_v = _normalize(box.world_transform.rotate_vector((0.0, 1.0, 0.0)))
    top_center = _add(box.world_transform.position, _scale(_normalize(normal), sz / 2.0))
    offset = _subtract(sphere.world_transform.position, top_center)
    u = _dot(offset, axis_u)
    v = _dot(offset, axis_v)
    if abs(u) > half_u + sphere.geometry.radius or abs(v) > half_v + sphere.geometry.radius:
        return False, "sphere initial projection is outside finite box top", None
    return True, "ok", AnalyticPlane(point=top_center, normal=normal, linear_velocity=(0.0, 0.0, 0.0))


def _metadata_sort_key(item: CompiledColliderMetadata) -> tuple[str, str, str]:
    return (item.runtime_body_id, item.collider_id, item.mujoco_geom_name)


def _canonical_pair_key(first: CompiledColliderMetadata, second: CompiledColliderMetadata) -> tuple[str, str]:
    a = f"{first.runtime_body_id}/{first.collider_id}/{first.mujoco_geom_name}"
    b = f"{second.runtime_body_id}/{second.collider_id}/{second.mujoco_geom_name}"
    return (a, b) if a <= b else (b, a)


def _canonical_metadata_pair(
    first: CompiledColliderMetadata,
    second: CompiledColliderMetadata,
) -> tuple[CompiledColliderMetadata, CompiledColliderMetadata]:
    return (first, second) if _metadata_sort_key(first) <= _metadata_sort_key(second) else (second, first)


def _candidate_id(kind: str, first: CompiledColliderMetadata, second: CompiledColliderMetadata) -> str:
    a, b = _canonical_pair_key(first, second)
    digest = hashlib.sha256(f"{kind}\0{a}\0{b}".encode("utf8")).hexdigest()[:10]
    return f"adaptive:{kind}:{_safe_id(a)}:{_safe_id(b)}:{digest}"


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")[:48] or "id"


def _candidate_type_name(candidate: AdaptiveCollisionCandidate) -> str:
    if isinstance(candidate, SpherePlaneAdaptiveCandidate):
        return "sphere_plane"
    if isinstance(candidate, SphereSphereAdaptiveCandidate):
        return "sphere_sphere"
    return type(candidate).__name__


def _diagnostic(
    first: CompiledColliderMetadata,
    second: CompiledColliderMetadata,
    status: AdaptiveCandidateDiagnosticStatus,
    reason: str,
    generated_candidate_id: str | None,
) -> AdaptiveCandidateDiagnostic:
    ordered_first, ordered_second = _canonical_metadata_pair(first, second)
    return AdaptiveCandidateDiagnostic(
        collider_a_id=ordered_first.collider_id,
        collider_b_id=ordered_second.collider_id,
        runtime_body_a_id=ordered_first.runtime_body_id,
        runtime_body_b_id=ordered_second.runtime_body_id,
        geometry_a_type=type(ordered_first.geometry).__name__,
        geometry_b_type=type(ordered_second.geometry).__name__,
        status=status,
        reason=reason,
        generated_candidate_id=generated_candidate_id,
    )


def _add(first: Vector3, second: Vector3) -> Vector3:
    return tuple(first[index] + second[index] for index in range(3))  # type: ignore[return-value]


def _subtract(first: Vector3, second: Vector3) -> Vector3:
    return tuple(first[index] - second[index] for index in range(3))  # type: ignore[return-value]


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return tuple(value * scalar for value in vector)  # type: ignore[return-value]


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(first[index] * second[index] for index in range(3))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector3) -> Vector3:
    length = _norm(vector)
    if length <= 1.0e-12:
        return (0.0, 0.0, 1.0)
    return _scale(vector, 1.0 / length)

