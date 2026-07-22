"""MuJoCo-specific optional configuration types."""

from physical_simulation.mujoco.collision_prediction import (
    AnalyticPlane,
    CollisionPrediction,
    SolverCollisionEstimate,
    estimate_solver_collision,
    predict_sphere_plane_collision,
    predict_sphere_sphere_collision,
)
from physical_simulation.mujoco.contact_params import (
    DEFAULT_MUJOCO_CONTACT_SOLVER_PARAMS,
    MuJoCoContactSolverParams,
    resolve_mujoco_contact_solver_params,
)
from physical_simulation.mujoco.contact_timescale import (
    DampingRegime,
    SolverContactTimescale,
    SubstepRecommendation,
    SubstepRecommendationConfig,
    estimate_solver_contact_timescale,
    recommend_solver_substeps,
)

__all__ = [
    "AnalyticPlane",
    "AdaptiveMuJoCoRunner",
    "AdaptiveCandidateBuildConfig",
    "AdaptiveCandidateBuildResult",
    "AdaptiveCandidateDiagnostic",
    "AdaptiveCandidateDiagnosticStatus",
    "AdaptiveStepDecision",
    "AdaptiveStepResult",
    "AdaptiveSubstepConfig",
    "ContactMotionState",
    "CollisionPrediction",
    "DampingRegime",
    "DEFAULT_MUJOCO_CONTACT_SOLVER_PARAMS",
    "MuJoCoContactSolverParams",
    "MuJoCoSubstepRunner",
    "SolverCollisionEstimate",
    "SolverContactTimescale",
    "SpherePlaneAdaptiveCandidate",
    "SphereSphereAdaptiveCandidate",
    "SubstepAdvanceResult",
    "SubstepRecommendation",
    "SubstepRecommendationConfig",
    "build_adaptive_prediction_candidates",
    "create_adaptive_runner_from_scene",
    "estimate_solver_collision",
    "estimate_solver_contact_timescale",
    "predict_sphere_plane_collision",
    "predict_sphere_sphere_collision",
    "recommend_solver_substeps",
    "resolve_mujoco_contact_solver_params",
]


def __getattr__(name: str):
    if name in {"MuJoCoSubstepRunner", "SubstepAdvanceResult"}:
        from physical_simulation.mujoco.substepping import MuJoCoSubstepRunner, SubstepAdvanceResult

        values = {
            "MuJoCoSubstepRunner": MuJoCoSubstepRunner,
            "SubstepAdvanceResult": SubstepAdvanceResult,
        }
        return values[name]
    if name in {
        "AdaptiveCandidateBuildConfig",
        "AdaptiveCandidateBuildResult",
        "AdaptiveCandidateDiagnostic",
        "AdaptiveCandidateDiagnosticStatus",
        "build_adaptive_prediction_candidates",
        "create_adaptive_runner_from_scene",
    }:
        from physical_simulation.mujoco.adaptive_candidates import (
            AdaptiveCandidateBuildConfig,
            AdaptiveCandidateBuildResult,
            AdaptiveCandidateDiagnostic,
            AdaptiveCandidateDiagnosticStatus,
            build_adaptive_prediction_candidates,
            create_adaptive_runner_from_scene,
        )

        values = {
            "AdaptiveCandidateBuildConfig": AdaptiveCandidateBuildConfig,
            "AdaptiveCandidateBuildResult": AdaptiveCandidateBuildResult,
            "AdaptiveCandidateDiagnostic": AdaptiveCandidateDiagnostic,
            "AdaptiveCandidateDiagnosticStatus": AdaptiveCandidateDiagnosticStatus,
            "build_adaptive_prediction_candidates": build_adaptive_prediction_candidates,
            "create_adaptive_runner_from_scene": create_adaptive_runner_from_scene,
        }
        return values[name]
    if name in {
        "AdaptiveMuJoCoRunner",
        "AdaptiveStepDecision",
        "AdaptiveStepResult",
        "AdaptiveSubstepConfig",
        "ContactMotionState",
        "SpherePlaneAdaptiveCandidate",
        "SphereSphereAdaptiveCandidate",
    }:
        from physical_simulation.mujoco.adaptive import (
            AdaptiveMuJoCoRunner,
            AdaptiveStepDecision,
            AdaptiveStepResult,
            AdaptiveSubstepConfig,
            ContactMotionState,
            SpherePlaneAdaptiveCandidate,
            SphereSphereAdaptiveCandidate,
        )

        values = {
            "AdaptiveMuJoCoRunner": AdaptiveMuJoCoRunner,
            "AdaptiveStepDecision": AdaptiveStepDecision,
            "AdaptiveStepResult": AdaptiveStepResult,
            "AdaptiveSubstepConfig": AdaptiveSubstepConfig,
            "ContactMotionState": ContactMotionState,
            "SpherePlaneAdaptiveCandidate": SpherePlaneAdaptiveCandidate,
            "SphereSphereAdaptiveCandidate": SphereSphereAdaptiveCandidate,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
