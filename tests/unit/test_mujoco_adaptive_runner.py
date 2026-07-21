import pytest

from physical_simulation.backends import BackendNotLoadedError, MuJoCoBackend
from physical_simulation.mujoco import (
    AdaptiveSubstepConfig,
    AnalyticPlane,
    ContactMotionState,
    MuJoCoContactSolverParams,
    SpherePlaneAdaptiveCandidate,
    SphereSphereAdaptiveCandidate,
    SubstepRecommendationConfig,
)
from physical_simulation.validation.errors import PhysicsValidationError


def _params(solref=(0.005, 0.3)):
    return MuJoCoContactSolverParams(solref=solref, solimp=(0.9, 0.9, 0.001, 0.5, 2.0))


def test_adaptive_config_validation() -> None:
    config = AdaptiveSubstepConfig(
        macro_timestep=1.0 / 240.0,
        recommendation=SubstepRecommendationConfig(maximum_substeps=16),
    )

    assert config.macro_timestep == pytest.approx(1.0 / 240.0)
    with pytest.raises(PhysicsValidationError, match="macro_timestep"):
        AdaptiveSubstepConfig(macro_timestep=0.0)
    with pytest.raises(PhysicsValidationError, match="prediction_horizon_multiplier"):
        AdaptiveSubstepConfig(prediction_horizon_multiplier=float("inf"))
    with pytest.raises(PhysicsValidationError, match="resting_window_macro_steps"):
        AdaptiveSubstepConfig(resting_window_macro_steps=0)


def test_adaptive_candidate_validation() -> None:
    plane = AnalyticPlane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))

    candidate = SpherePlaneAdaptiveCandidate("ground", "sphere", 0.1, plane, _params())

    assert candidate.candidate_id == "ground"
    with pytest.raises(PhysicsValidationError, match="sphere_radius"):
        SpherePlaneAdaptiveCandidate("bad", "sphere", 0.0, plane, _params())
    with pytest.raises(PhysicsValidationError, match="different"):
        SphereSphereAdaptiveCandidate("bad", "a", 0.1, "a", 0.1, _params())


def test_contact_motion_state_values_are_stable() -> None:
    assert [state.value for state in ContactMotionState] == [
        "free",
        "approaching",
        "impacting",
        "resting",
        "separating",
    ]


def test_adaptive_runner_requires_loaded_backend() -> None:
    from physical_simulation.mujoco import AdaptiveMuJoCoRunner

    with pytest.raises(BackendNotLoadedError):
        AdaptiveMuJoCoRunner(
            MuJoCoBackend(),
            candidates=(),
            config=AdaptiveSubstepConfig(),
        )
