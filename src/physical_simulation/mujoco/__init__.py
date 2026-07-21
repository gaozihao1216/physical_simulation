"""MuJoCo-specific optional configuration types."""

from physical_simulation.mujoco.contact_params import MuJoCoContactSolverParams

__all__ = ["MuJoCoContactSolverParams", "MuJoCoSubstepRunner", "SubstepAdvanceResult"]


def __getattr__(name: str):
    if name in {"MuJoCoSubstepRunner", "SubstepAdvanceResult"}:
        from physical_simulation.mujoco.substepping import MuJoCoSubstepRunner, SubstepAdvanceResult

        values = {
            "MuJoCoSubstepRunner": MuJoCoSubstepRunner,
            "SubstepAdvanceResult": SubstepAdvanceResult,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
