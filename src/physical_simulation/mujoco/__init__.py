"""MuJoCo-specific optional configuration types."""

from physical_simulation.mujoco.collision_prediction import (
    AnalyticPlane,
    CollisionPrediction,
    SolverCollisionEstimate,
    estimate_solver_collision,
    predict_sphere_plane_collision,
    predict_sphere_sphere_collision,
)
from physical_simulation.mujoco.contact_params import MuJoCoContactSolverParams
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
    "CollisionPrediction",
    "DampingRegime",
    "MuJoCoContactSolverParams",
    "MuJoCoSubstepRunner",
    "SolverCollisionEstimate",
    "SolverContactTimescale",
    "SubstepAdvanceResult",
    "SubstepRecommendation",
    "SubstepRecommendationConfig",
    "estimate_solver_collision",
    "estimate_solver_contact_timescale",
    "predict_sphere_plane_collision",
    "predict_sphere_sphere_collision",
    "recommend_solver_substeps",
]


def __getattr__(name: str):
    if name in {"MuJoCoSubstepRunner", "SubstepAdvanceResult"}:
        from physical_simulation.mujoco.substepping import MuJoCoSubstepRunner, SubstepAdvanceResult

        values = {
            "MuJoCoSubstepRunner": MuJoCoSubstepRunner,
            "SubstepAdvanceResult": SubstepAdvanceResult,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
