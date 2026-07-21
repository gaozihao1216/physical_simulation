from xml.etree import ElementTree as ET

import pytest

from physical_simulation.assets import (
    ColliderSpec,
    ConeGeometry,
    EllipsoidGeometry,
    FrustumGeometry,
    RegularPrismGeometry,
    RigidBodySpec,
    SphericalCapGeometry,
    Transform,
    VisualSpec,
    WedgeGeometry,
    create_single_body_asset,
)
from physical_simulation.compilers import MuJoCoCompiler, UnsupportedPhysicsFeatureError
from physical_simulation.compilers.mujoco_mesh import geometry_to_convex_mesh
from physical_simulation.scene import AssetInstanceSpec, create_scene


def _compile_static_geometry(geometry):
    body = RigidBodySpec(
        "body",
        "body",
        "static",
        Transform.identity(),
        (),
        (ColliderSpec("collider", geometry),),
    )
    asset = create_single_body_asset(asset_id="asset", body=body)
    scene = create_scene(scene_id="mesh_fallback", instances=(AssetInstanceSpec("inst", asset),))
    result = MuJoCoCompiler().compile(scene)
    return result, ET.fromstring(result.mjcf)


@pytest.mark.parametrize(
    ("geometry", "expected_vertex_count"),
    [
        (WedgeGeometry((1.0, 2.0, 3.0)), 6),
        (ConeGeometry(0.5, 2.0), 34),
        (FrustumGeometry(0.5, 0.25, 2.0), 66),
        (RegularPrismGeometry(5, 0.5, 2.0), 12),
    ],
)
def test_convex_mesh_fallback_generates_vertices_and_faces(geometry, expected_vertex_count) -> None:
    mesh = geometry_to_convex_mesh(geometry)

    assert len(mesh.vertices) == expected_vertex_count
    assert len(mesh.faces) > 0
    assert mesh.vertex_attribute()
    assert mesh.face_attribute()


@pytest.mark.parametrize(
    "geometry",
    [
        WedgeGeometry((1.0, 2.0, 3.0)),
        ConeGeometry(0.5, 2.0),
        FrustumGeometry(0.5, 0.25, 2.0),
        RegularPrismGeometry(5, 0.5, 2.0),
    ],
)
def test_compiler_emits_mesh_asset_for_supported_fallback_geometry(geometry) -> None:
    _result, root = _compile_static_geometry(geometry)

    meshes = root.findall("./asset/mesh")
    geoms = root.findall(".//geom")

    assert len(meshes) == 1
    assert len(geoms) == 1
    assert geoms[0].attrib["type"] == "mesh"
    assert geoms[0].attrib["mesh"] == meshes[0].attrib["name"]
    assert "size" not in geoms[0].attrib


def test_compiler_reuses_identical_mesh_assets() -> None:
    geometry = WedgeGeometry((1.0, 2.0, 3.0))
    body = RigidBodySpec(
        "body",
        "body",
        "static",
        Transform.identity(),
        (),
        (
            ColliderSpec("first", geometry),
            ColliderSpec("second", geometry, Transform(position=(1.0, 0.0, 0.0))),
        ),
    )
    asset = create_single_body_asset(asset_id="asset", body=body)
    scene = create_scene(scene_id="mesh_reuse", instances=(AssetInstanceSpec("inst", asset),))
    root = ET.fromstring(MuJoCoCompiler().compile(scene).mjcf)

    meshes = root.findall("./asset/mesh")
    geoms = root.findall(".//geom")

    assert len(meshes) == 1
    assert {geom.attrib["mesh"] for geom in geoms} == {meshes[0].attrib["name"]}


def test_visual_mesh_fallback_keeps_collision_disabled() -> None:
    geometry = WedgeGeometry((1.0, 2.0, 3.0))
    body = RigidBodySpec(
        "body",
        "body",
        "static",
        Transform.identity(),
        (VisualSpec("visual", geometry),),
        (),
    )
    asset = create_single_body_asset(asset_id="asset", body=body)
    scene = create_scene(scene_id="visual_mesh", instances=(AssetInstanceSpec("inst", asset),))
    root = ET.fromstring(MuJoCoCompiler().compile(scene).mjcf)
    geom = root.find(".//geom")

    assert geom.attrib["type"] == "mesh"
    assert geom.attrib["contype"] == "0"
    assert geom.attrib["conaffinity"] == "0"


@pytest.mark.parametrize("geometry", [EllipsoidGeometry((1.0, 2.0, 3.0)), SphericalCapGeometry(1.0, 0.5)])
def test_curved_geometry_without_mesh_fallback_still_raises(geometry) -> None:
    with pytest.raises(UnsupportedPhysicsFeatureError):
        _compile_static_geometry(geometry)
