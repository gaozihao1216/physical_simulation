from __future__ import annotations

import math

import pytest

mujoco = pytest.importorskip("mujoco")

from physical_simulation.assets import (
    BoxGeometry,
    ColliderSpec,
    MassProperties,
    PhysicsAssetSpec,
    RigidBodySpec,
    Transform,
    VisualSpec,
    create_box,
    create_ground,
    create_single_body_asset,
)
from physical_simulation.backends import (
    BackendNotLoadedError,
    MuJoCoBackend,
    MuJoCoModelLoadingError,
    UnknownRuntimeBodyError,
)
from physical_simulation.compilers import MuJoCoCompilationResult
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _box_drop_scene():
    ground_asset = create_single_body_asset(
        asset_id="ground_asset",
        body=create_ground(body_id="ground_body"),
    )
    box_body = create_box("box_body", (0.4, 0.4, 0.4), mass=1.0)
    box_asset = create_single_body_asset(asset_id="box_asset", body=box_body)
    scene = create_scene(
        scene_id="box_drop",
        instances=(
            AssetInstanceSpec("ground_01", ground_asset, Transform.identity(), fixed_base=True),
            AssetInstanceSpec("box_01", box_asset, Transform(position=(0.0, 0.0, 1.0))),
        ),
        gravity=(0.0, 0.0, -9.81),
        timestep=1.0 / 240.0,
    )
    return scene, box_body


def test_load_scene_creates_model_data_and_preserves_scene_parameters() -> None:
    scene, _ = _box_drop_scene()
    backend = MuJoCoBackend()
    backend.load_scene(scene)

    assert backend.is_loaded is True
    assert backend.scene == scene
    assert backend.compilation_result is not None
    assert "<mujoco" in backend.mjcf
    assert backend._model is not None
    assert backend._data is not None
    assert backend._model.opt.timestep == pytest.approx(scene.timestep)
    assert tuple(float(value) for value in backend._model.opt.gravity) == pytest.approx(scene.gravity)


def test_runtime_body_numeric_id_mapping_and_unknown_body() -> None:
    scene, _ = _box_drop_scene()
    backend = MuJoCoBackend()
    backend.load_scene(scene)

    ground_id = backend._runtime_body_to_mj_body_id["ground_01/ground_body"]
    box_id = backend._runtime_body_to_mj_body_id["box_01/box_body"]
    assert ground_id != box_id
    assert ground_id != 0
    assert box_id != 0
    assert backend._mj_body_id_to_runtime_body[ground_id] == "ground_01/ground_body"
    assert backend._mj_body_id_to_runtime_body[box_id] == "box_01/box_body"
    with pytest.raises(UnknownRuntimeBodyError, match="missing/body"):
        backend._require_runtime_body_id("missing/body")


def test_dynamic_mass_and_inertia_match_physics_ir() -> None:
    scene, box_body = _box_drop_scene()
    backend = MuJoCoBackend()
    backend.load_scene(scene)

    body_id = backend._runtime_body_to_mj_body_id["box_01/box_body"]
    assert box_body.mass_properties is not None
    assert float(backend._model.body_mass[body_id]) == pytest.approx(box_body.mass_properties.mass)
    assert tuple(float(value) for value in backend._model.body_inertia[body_id]) == pytest.approx(
        box_body.mass_properties.inertia_diagonal
    )


def test_dynamic_principal_axes_orientation_loads_into_mujoco_inertial_frame() -> None:
    half = math.sqrt(0.5)
    mass_properties = MassProperties.from_principal_axes(
        mass=2.0,
        center_of_mass=(0.1, 0.2, 0.3),
        principal_inertia=(1.0, 2.0, 3.0),
        principal_axes=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    body = RigidBodySpec(
        "body",
        "body",
        "dynamic",
        Transform.identity(),
        (),
        (ColliderSpec("collider", BoxGeometry((1.0, 1.0, 1.0))),),
        mass_properties,
    )
    asset = create_single_body_asset(asset_id="asset", body=body)
    scene = create_scene(scene_id="principal_axes", instances=(AssetInstanceSpec("inst", asset),))
    backend = MuJoCoBackend()
    backend.load_scene(scene)

    body_id = backend._runtime_body_to_mj_body_id["inst/body"]
    assert tuple(float(value) for value in backend._model.body_inertia[body_id]) == pytest.approx((1.0, 2.0, 3.0))
    assert tuple(float(value) for value in backend._model.body_iquat[body_id]) == pytest.approx(
        (half, 0.0, 0.0, half)
    )


def test_compound_collider_mapping_excludes_visual_geoms() -> None:
    geometry = BoxGeometry((1.0, 1.0, 1.0))
    body = RigidBodySpec(
        "table_body",
        "table_body",
        "static",
        Transform.identity(),
        (VisualSpec("visual", geometry),),
        (
            ColliderSpec("top", geometry),
            ColliderSpec("leg", geometry, Transform(position=(1.0, 0.0, 0.0))),
        ),
    )
    asset = create_single_body_asset(asset_id="table_asset", body=body)
    scene = create_scene(scene_id="compound", instances=(AssetInstanceSpec("table_01", asset),))
    backend = MuJoCoBackend()
    backend.load_scene(scene)

    runtime_id = "table_01/table_body"
    assert len(backend._runtime_body_to_mj_body_id) == 1
    assert len(backend._runtime_body_to_collision_geom_ids[runtime_id]) == 2
    assert len(backend._mj_geom_id_to_runtime_body) == 2
    assert set(backend._mj_geom_id_to_runtime_body.values()) == {runtime_id}


def test_multiple_instances_have_distinct_body_and_geom_ids() -> None:
    box_asset = create_single_body_asset(
        asset_id="box_asset",
        body=create_box("box_body", (0.4, 0.4, 0.4), mass=1.0),
    )
    scene = create_scene(
        scene_id="multi",
        instances=(
            AssetInstanceSpec("box_01", box_asset, Transform(position=(-1.0, 0.0, 1.0))),
            AssetInstanceSpec("box_02", box_asset, Transform(position=(1.0, 0.0, 1.0))),
        ),
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)

    first_body = backend._runtime_body_to_mj_body_id["box_01/box_body"]
    second_body = backend._runtime_body_to_mj_body_id["box_02/box_body"]
    first_geoms = backend._runtime_body_to_collision_geom_ids["box_01/box_body"]
    second_geoms = backend._runtime_body_to_collision_geom_ids["box_02/box_body"]
    assert first_body != second_body
    assert set(first_geoms).isdisjoint(second_geoms)
    assert all(backend._mj_geom_id_to_runtime_body[geom_id] == "box_01/box_body" for geom_id in first_geoms)
    assert all(backend._mj_geom_id_to_runtime_body[geom_id] == "box_02/box_body" for geom_id in second_geoms)


def test_static_dynamic_fixed_and_kinematic_load_with_expected_joints() -> None:
    dynamic = create_single_body_asset(asset_id="dynamic_asset", body=create_box("dynamic_body", (1.0, 1.0, 1.0), mass=1.0))
    fixed_dynamic = create_single_body_asset(asset_id="fixed_dynamic_asset", body=create_box("fixed_dynamic_body", (1.0, 1.0, 1.0), mass=1.0))
    static = create_single_body_asset(asset_id="static_asset", body=create_ground(body_id="static_body"))
    kinematic_body = RigidBodySpec(
        "kinematic_body",
        "kinematic_body",
        "kinematic",
        Transform.identity(),
        (),
        (ColliderSpec("kinematic_collider", BoxGeometry((1.0, 1.0, 1.0))),),
    )
    kinematic = create_single_body_asset(asset_id="kinematic_asset", body=kinematic_body)
    scene = create_scene(
        scene_id="body_types",
        instances=(
            AssetInstanceSpec("static", static),
            AssetInstanceSpec("dynamic", dynamic, Transform(position=(2.0, 0.0, 0.0))),
            AssetInstanceSpec("fixed_dynamic", fixed_dynamic, Transform(position=(4.0, 0.0, 0.0)), fixed_base=True),
            AssetInstanceSpec("kinematic", kinematic, Transform(position=(6.0, 0.0, 0.0))),
        ),
    )
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    assert backend._model.njnt == 1
    assert len(backend._runtime_body_to_mj_body_id) == 4


def test_default_collision_mask_loads_in_real_mujoco() -> None:
    scene, _ = _box_drop_scene()
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    for geom_id in backend._mj_geom_id_to_runtime_body:
        assert int(backend._model.geom_conaffinity[geom_id]) == 2147483647


class _BadCompiler:
    def compile(self, scene):
        return MuJoCoCompilationResult(
            scene_id=scene.scene_id,
            mjcf="<mujoco><worldbody><body></worldbody></mujoco>",
            runtime_body_to_mujoco_name=(),
            mujoco_geom_to_runtime_body=(),
        )


def test_failed_loading_keeps_previous_valid_scene() -> None:
    scene, _ = _box_drop_scene()
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    previous_mjcf = backend.mjcf

    backend._compiler = _BadCompiler()
    with pytest.raises(MuJoCoModelLoadingError):
        backend.load_scene(scene)
    assert backend.is_loaded
    assert backend.scene == scene
    assert backend.mjcf == previous_mjcf


def test_close_clears_state_and_allows_reload() -> None:
    scene, _ = _box_drop_scene()
    backend = MuJoCoBackend()
    backend.load_scene(scene)
    backend.close()

    assert not backend.is_loaded
    assert backend.scene is None
    assert backend.compilation_result is None
    assert backend._model is None
    assert backend._data is None
    assert backend._runtime_body_to_mj_body_id == {}
    with pytest.raises(BackendNotLoadedError):
        _ = backend.mjcf
    backend.close()
    backend.load_scene(scene)
    assert backend.is_loaded
