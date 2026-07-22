"""MuJoCo-specific contact solver parameter specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from physical_simulation.validation.asset_validator import _as_float_tuple, _finite_float
from physical_simulation.validation.errors import PhysicsValidationError


DEFAULT_MUJOCO_CONTACT_SOLVER_PARAMS = None


@dataclass(frozen=True)
class MuJoCoContactSolverParams:
    """Optional MuJoCo soft-contact parameters for one collision geom."""

    solref: tuple[float, float]
    solimp: tuple[float, float, float, float, float]
    margin: float = 0.0
    gap: float = 0.0
    priority: int = 0
    solmix: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "solref",
            _as_float_tuple(
                self.solref,
                field_name="solref",
                length=2,
                error_type=PhysicsValidationError,
            ),
        )
        object.__setattr__(
            self,
            "solimp",
            _as_float_tuple(
                self.solimp,
                field_name="solimp",
                length=5,
                error_type=PhysicsValidationError,
            ),
        )
        object.__setattr__(
            self,
            "margin",
            _finite_float(
                self.margin,
                field_name="margin",
                minimum=0.0,
                error_type=PhysicsValidationError,
            ),
        )
        object.__setattr__(
            self,
            "gap",
            _finite_float(
                self.gap,
                field_name="gap",
                minimum=0.0,
                error_type=PhysicsValidationError,
            ),
        )
        if not isinstance(self.priority, int) or isinstance(self.priority, bool) or self.priority < 0:
            raise PhysicsValidationError(f"priority must be a non-negative int; actual value={self.priority!r}")
        object.__setattr__(
            self,
            "solmix",
            _finite_float(
                self.solmix,
                field_name="solmix",
                minimum=0.0,
                error_type=PhysicsValidationError,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the MuJoCo contact parameters to a JSON-compatible dictionary."""
        return {
            "solref": list(self.solref),
            "solimp": list(self.solimp),
            "margin": self.margin,
            "gap": self.gap,
            "priority": self.priority,
            "solmix": self.solmix,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MuJoCoContactSolverParams":
        """Deserialize MuJoCo contact parameters from a dictionary."""
        if not isinstance(data, dict):
            raise PhysicsValidationError(f"MuJoCo contact solver params data must be a dict; actual value={data!r}")
        return cls(
            solref=tuple(data.get("solref", ())),
            solimp=tuple(data.get("solimp", ())),
            margin=data.get("margin", 0.0),
            gap=data.get("gap", 0.0),
            priority=data.get("priority", 0),
            solmix=data.get("solmix", 1.0),
        )


DEFAULT_MUJOCO_CONTACT_SOLVER_PARAMS = MuJoCoContactSolverParams(
    solref=(0.02, 1.0),
    solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
)


def resolve_mujoco_contact_solver_params(
    first: MuJoCoContactSolverParams | None,
    second: MuJoCoContactSolverParams | None,
    *,
    default: MuJoCoContactSolverParams = DEFAULT_MUJOCO_CONTACT_SOLVER_PARAMS,
) -> MuJoCoContactSolverParams:
    """Resolve a deterministic pair-level solver parameter estimate.

    This mirrors the project policy used for explicit fixed-fixed pairs. Dynamic
    contacts still use MuJoCo's native geom priority/solmix behavior at runtime;
    this helper only provides a stable solver-timescale estimate for adaptive
    pre-contact prediction.
    """
    if first is None and second is None:
        return default
    if first is None:
        return second  # type: ignore[return-value]
    if second is None:
        return first
    if first.priority > second.priority:
        return first
    if second.priority > first.priority:
        return second
    return MuJoCoContactSolverParams(
        solref=_mix_equal_priority_solref(first, second),
        solimp=_weighted_average(first.solimp, first.solmix, second.solimp, second.solmix),
        margin=first.margin + second.margin,
        gap=first.gap + second.gap,
        priority=first.priority,
        solmix=max(first.solmix, second.solmix),
    )


def _mix_equal_priority_solref(
    first: MuJoCoContactSolverParams,
    second: MuJoCoContactSolverParams,
) -> tuple[float, float]:
    return _weighted_average(first.solref, first.solmix, second.solref, second.solmix)


def _weighted_average(
    first: tuple[float, ...],
    first_weight: float,
    second: tuple[float, ...],
    second_weight: float,
) -> tuple[float, ...]:
    total = first_weight + second_weight
    if total <= 1.0e-12:
        return tuple((a + b) / 2.0 for a, b in zip(first, second))
    return tuple((a * first_weight + b * second_weight) / total for a, b in zip(first, second))
