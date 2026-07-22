"""Automatic adaptive prediction candidate construction tests."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from physical_simulation.assets import (
    ColliderSpec,
    Transform,
    create_box,
    create_capsule,
    create_single_body_asset,
    create_sphere,
)
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.mujoco.adaptive import _point_in_bounds
from physical_simulation.mujoco import (
    AdaptiveCandidateBuildConfig,
    ConservativePrimitiveAdaptiveCandidate,
    MuJoCoContactSolverParams,
    SpherePlaneAdaptiveCandidate,
    SphereSphereAdaptiveCandidate,
    build_adaptive_prediction_candidates,
)
from physical_simulation.scene import AssetInstanceSpec, create_scene


def test_builds_sphere_plane_candidate_for_dynamic_sphere_and_static_large_box() -> None:
    scene = _sphere_ground_scene()
    result = _build(scene)
    assert result.inspected_collider_count == 2
    assert result.eligible_pair_count == 1
    assert result.generated_candidate_count == 1
    candidate = result.candidates[0]
    assert isinstance(candidate, SpherePlaneAdaptiveCandidate)
    assert candidate.sphere_runtime_body_id == "sphere_01/sphere"
    assert candidate.sphere_radius == 0.1
    assert candidate.plane.normal == pytest.approx((0.0, 0.0, 1.0))
    assert candidate.finite_bounds is not None
    assert _point_in_bounds((0.0, 0.0, 0.0), candidate.finite_bounds)
    assert not _point_in_bounds((2.0, 0.0, 0.0), candidate.finite_bounds)


def test_builds_sphere_sphere_candidate_with_stable_order_and_contact_params() -> None:
    params = MuJoCoContactSolverParams(solref=(0.01, 0.3), solimp=(0.8, 0.9, 0.001, 0.5, 2.0), priority=2)
    scene = _sphere_sphere_scene(params=params)
    reversed_scene = create_scene(scene_id="ss_reversed", instances=tuple(reversed(scene.instances)), gravity=(0.0, 0.0, 0.0), timestep=scene.timestep)
    first = _build(scene)
    second = _build(reversed_scene)
    assert first.generated_candidate_count == 1
    candidate = first.candidates[0]
    assert isinstance(candidate, SphereSphereAdaptiveCandidate)
    assert candidate.body_a_id < candidate.body_b_id
    assert candidate.contact_params.solref == (0.01, 0.3)
    assert [type(c).__name__ for c in first.candidates] == [type(c).__name__ for c in second.candidates]
    assert first.candidates[0].candidate_id == second.candidates[0].candidate_id
    assert first.candidates[0].candidate_id != ""


def test_same_runtime_body_colliders_are_skipped() -> None:
    sphere = create_sphere("sphere", 0.1, mass=1.0, create_visual=False)
    extra = ColliderSpec(collider_id="extra", geometry=sphere.colliders[0].geometry, material_id=sphere.colliders[0].material_id)
    sphere = replace(sphere, colliders=(sphere.colliders[0], extra))
    scene = create_scene(
        scene_id="same_body",
        instances=(AssetInstanceSpec("sphere_01", create_single_body_asset(asset_id="sphere_asset", body=sphere)),),
        timestep=1.0 / 240.0,
    )
    result = _build(scene)
    assert result.generated_candidate_count == 0
    assert result.skipped_same_body_count == 1


def test_mask_incompatible_pair_is_skipped() -> None:
    scene = _sphere_ground_scene(mask=0, group=0)
    result = _build(scene)
    assert result.generated_candidate_count == 0
    assert result.skipped_mask_count == 1


def test_static_static_pair_is_skipped() -> None:
    box_a = create_single_body_asset(asset_id="a_asset", body=create_box("a", (1, 1, 1), body_type="static", create_visual=False))
    box_b = create_single_body_asset(asset_id="b_asset", body=create_box("b", (1, 1, 1), body_type="static", create_visual=False))
    scene = create_scene(
        scene_id="static_static",
        instances=(AssetInstanceSpec("a_01", box_a, fixed_base=True), AssetInstanceSpec("b_01", box_b, fixed_base=True)),
        timestep=1.0 / 240.0,
    )
    result = _build(scene)
    assert result.generated_candidate_count == 0
    assert result.skipped_static_static_count == 1


def test_visual_only_geom_is_not_inspected() -> None:
    sphere = create_sphere("sphere", 0.1, body_type="static", create_visual=True, create_collider=False)
    scene = create_scene(
        scene_id="visual_only",
        instances=(AssetInstanceSpec("sphere_01", create_single_body_asset(asset_id="sphere_asset", body=sphere)),),
        timestep=1.0 / 240.0,
    )
    result = _build(scene)
    assert result.inspected_collider_count == 0
    assert result.generated_candidate_count == 0


def test_tilted_or_too_small_box_top_is_skipped() -> None:
    tilted = _sphere_ground_scene(ground_transform=Transform(rotation=(math.cos(0.2), math.sin(0.2), 0.0, 0.0), position=(0, 0, -0.05)))
    small = _sphere_ground_scene(ground_size=(0.2, 0.2, 0.1))
    config = AdaptiveCandidateBuildConfig(include_conservative_primitives=False)
    assert _build(tilted, config=config).skipped_invalid_plane_count == 1
    assert _build(small, config=config).skipped_invalid_plane_count == 1
    assert isinstance(_build(tilted).candidates[0], ConservativePrimitiveAdaptiveCandidate)
    assert isinstance(_build(small).candidates[0], ConservativePrimitiveAdaptiveCandidate)


def test_unsupported_geometry_uses_conservative_fallback() -> None:
    capsule = create_single_body_asset(asset_id="capsule_asset", body=create_capsule("capsule", 0.05, 0.2, mass=1.0, create_visual=False))
    box = create_single_body_asset(asset_id="box_asset", body=create_box("box", (1, 1, 0.1), body_type="static", create_visual=False))
    scene = create_scene(
        scene_id="unsupported",
        instances=(AssetInstanceSpec("capsule_01", capsule), AssetInstanceSpec("box_01", box, fixed_base=True)),
        timestep=1.0 / 240.0,
    )
    result = _build(scene)
    assert result.generated_candidate_count == 1
    assert result.skipped_unsupported_geometry_count == 0
    assert isinstance(result.candidates[0], ConservativePrimitiveAdaptiveCandidate)


def test_box_box_uses_conservative_candidate() -> None:
    scene = _box_box_scene()
    result = _build(scene)
    assert result.generated_candidate_count == 1
    candidate = result.candidates[0]
    assert isinstance(candidate, ConservativePrimitiveAdaptiveCandidate)
    assert candidate.bounding_radius_a > 0.0
    assert candidate.bounding_radius_b > 0.0


def test_compound_body_pairs_are_aggregated_by_runtime_body_pair() -> None:
    first = create_box("body_a", (0.2, 0.2, 0.2), mass=1.0, create_visual=False)
    second = create_box("body_b", (0.2, 0.2, 0.2), mass=1.0, create_visual=False)
    first = replace(
        first,
        colliders=(
            first.colliders[0],
            replace(first.colliders[0], collider_id="body_a_offset", local_transform=Transform(position=(0.3, 0.0, 0.0))),
        ),
    )
    second = replace(
        second,
        colliders=(
            second.colliders[0],
            replace(second.colliders[0], collider_id="body_b_offset", local_transform=Transform(position=(-0.3, 0.0, 0.0))),
        ),
    )
    scene = create_scene(
        scene_id="compound_box_pair",
        instances=(
            AssetInstanceSpec("a_01", create_single_body_asset(asset_id="a_asset", body=first), Transform(position=(-0.6, 0, 0))),
            AssetInstanceSpec("b_01", create_single_body_asset(asset_id="b_asset", body=second), Transform(position=(0.6, 0, 0))),
        ),
        gravity=(0, 0, 0),
        timestep=1.0 / 240.0,
    )
    result = _build(scene)
    assert result.generated_candidate_count == 1
    assert result.eligible_pair_count == 4
    candidate = result.candidates[0]
    assert isinstance(candidate, ConservativePrimitiveAdaptiveCandidate)
    assert candidate.bounding_radius_a > 0.3
    assert candidate.bounding_radius_b > 0.3


def _build(scene, config=None):
    backend = MuJoCoBackend()
    try:
        backend.load_scene(scene)
        return build_adaptive_prediction_candidates(scene=scene, backend=backend, config=config)
    finally:
        backend.close()


def _sphere_ground_scene(*, mask: int = -1, group: int = 1, ground_size=(2.0, 2.0, 0.1), ground_transform=None):
    params = MuJoCoContactSolverParams(solref=(0.02, 0.5), solimp=(0.9, 0.95, 0.001, 0.5, 2.0))
    sphere = create_sphere("sphere", 0.1, mass=1.0, create_visual=False)
    sphere = replace(
        sphere,
        colliders=(replace(sphere.colliders[0], collision_group=group, collision_mask=mask, mujoco_contact_params=params),),
    )
    ground = create_box("ground", ground_size, body_type="static", transform=ground_transform or Transform(position=(0, 0, -ground_size[2] / 2)), create_visual=False)
    ground = replace(
        ground,
        colliders=(replace(ground.colliders[0], mujoco_contact_params=params),),
    )
    return create_scene(
        scene_id="sphere_ground",
        instances=(
            AssetInstanceSpec("ground_01", create_single_body_asset(asset_id="ground_asset", body=ground), fixed_base=True),
            AssetInstanceSpec("sphere_01", create_single_body_asset(asset_id="sphere_asset", body=sphere), Transform(position=(0.0, 0.0, 1.0))),
        ),
        timestep=1.0 / 240.0,
    )


def _sphere_sphere_scene(*, params):
    first = create_sphere("sphere_a", 0.1, mass=1.0, create_visual=False)
    second = create_sphere("sphere_b", 0.2, mass=1.0, create_visual=False)
    first = replace(first, colliders=(replace(first.colliders[0], mujoco_contact_params=params),))
    second = replace(second, colliders=(replace(second.colliders[0], mujoco_contact_params=params),))
    return create_scene(
        scene_id="sphere_sphere",
        instances=(
            AssetInstanceSpec("a_01", create_single_body_asset(asset_id="a_asset", body=first), Transform(position=(-0.5, 0, 0))),
            AssetInstanceSpec("b_01", create_single_body_asset(asset_id="b_asset", body=second), Transform(position=(0.5, 0, 0))),
        ),
        gravity=(0, 0, 0),
        timestep=1.0 / 240.0,
    )


def _box_box_scene():
    first = create_box("box_a", (0.2, 0.2, 0.2), mass=1.0, create_visual=False)
    second = create_box("box_b", (0.2, 0.2, 0.2), body_type="static", create_visual=False)
    return create_scene(
        scene_id="box_box",
        instances=(
            AssetInstanceSpec("a_01", create_single_body_asset(asset_id="a_asset", body=first), Transform(position=(0.0, 0.0, 0.8))),
            AssetInstanceSpec("b_01", create_single_body_asset(asset_id="b_asset", body=second), Transform(position=(0.0, 0.0, 0.0)), fixed_base=True),
        ),
        timestep=1.0 / 240.0,
    )
