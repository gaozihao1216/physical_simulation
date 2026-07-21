import pytest

from physical_simulation.assets import (
    ColliderSpec,
    ConeGeometry,
    FrustumGeometry,
    RegularPrismGeometry,
    RigidBodySpec,
    Transform,
    WedgeGeometry,
    create_single_body_asset,
)
from physical_simulation.backends import MuJoCoBackend
from physical_simulation.scene import AssetInstanceSpec, create_scene


@pytest.mark.parametrize(
    "geometry",
    [
        WedgeGeometry((1.0, 2.0, 0.5)),
        ConeGeometry(0.5, 1.0),
        FrustumGeometry(0.5, 0.25, 1.0),
        RegularPrismGeometry(6, 0.5, 1.0),
    ],
)
def test_mujoco_loads_convex_mesh_fallback_geometry(geometry) -> None:
    body = RigidBodySpec(
        "body",
        "body",
        "static",
        Transform.identity(),
        (),
        (ColliderSpec("collider", geometry),),
    )
    asset = create_single_body_asset(asset_id="asset", body=body)
    scene = create_scene(scene_id="mesh_load", instances=(AssetInstanceSpec("inst", asset),))
    backend = MuJoCoBackend()

    try:
        backend.load_scene(scene)
        assert "<mesh" in backend.mjcf
        assert backend._model.nmesh == 1
        assert backend._model.ngeom == 1
    finally:
        backend.close()
