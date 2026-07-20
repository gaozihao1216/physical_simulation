import builtins

import pytest

from physical_simulation.assets import BoxGeometry
from physical_simulation.backends import MuJoCoBackend, MuJoCoUnavailableError
from physical_simulation.backends.mujoco_backend import _import_mujoco
from physical_simulation.compilers import geometry_to_mujoco


def test_core_packages_import_without_mujoco_requirement() -> None:
    from physical_simulation.assets import Transform
    from physical_simulation.compilers import MuJoCoCompiler
    from physical_simulation.runtime import RigidBodyState
    from physical_simulation.scene import create_scene

    assert Transform.identity() is not None
    assert MuJoCoCompiler is not None
    assert RigidBodyState is not None
    assert create_scene is not None
    assert geometry_to_mujoco(BoxGeometry((1.0, 1.0, 1.0))) == ("box", (0.5, 0.5, 0.5))


def test_backend_can_be_instantiated_without_importing_mujoco() -> None:
    backend = MuJoCoBackend()
    assert not backend.is_loaded


def test_missing_mujoco_raises_project_error(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mujoco":
            raise ImportError("missing on purpose")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(MuJoCoUnavailableError, match=r"\.\[mujoco\]"):
        _import_mujoco()


def test_non_import_runtime_errors_are_not_dependency_errors(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mujoco":
            raise RuntimeError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="boom"):
        _import_mujoco()
