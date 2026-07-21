"""MuJoCo contact solver timescale estimates and substep recommendations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from physical_simulation.mujoco.contact_params import MuJoCoContactSolverParams
from physical_simulation.validation.asset_validator import _finite_float
from physical_simulation.validation.errors import PhysicsValidationError


class DampingRegime(Enum):
    """Second-order solver contact damping regime."""

    UNDERDAMPED = "underdamped"
    CRITICAL = "critical"
    OVERDAMPED = "overdamped"


@dataclass(frozen=True)
class SolverContactTimescale:
    """Estimated MuJoCo soft-contact constraint timescale."""

    solref_format: str
    assumed_impedance: float
    impedance_width_value: float
    effective_damping: float
    effective_stiffness: float
    natural_frequency: float
    damping_ratio: float
    regime: DampingRegime
    damped_frequency: float | None
    oscillatory_contact_duration: float | None
    fastest_mode_timescale: float
    characteristic_timescale: float


@dataclass(frozen=True)
class SubstepRecommendationConfig:
    """Configuration for solver-timescale-based fixed substep recommendations."""

    samples_per_characteristic_time: int = 16
    maximum_substeps: int = 128
    minimum_substep_timestep: float = 1.0e-6

    def __post_init__(self) -> None:
        if (
            not isinstance(self.samples_per_characteristic_time, int)
            or isinstance(self.samples_per_characteristic_time, bool)
            or self.samples_per_characteristic_time < 1
        ):
            raise PhysicsValidationError(
                "samples_per_characteristic_time must be an int >= 1; "
                f"actual value={self.samples_per_characteristic_time!r}"
            )
        if not isinstance(self.maximum_substeps, int) or isinstance(self.maximum_substeps, bool) or self.maximum_substeps < 1:
            raise PhysicsValidationError(f"maximum_substeps must be an int >= 1; actual value={self.maximum_substeps!r}")
        object.__setattr__(
            self,
            "minimum_substep_timestep",
            _finite_float(
                self.minimum_substep_timestep,
                field_name="minimum_substep_timestep",
                minimum=0.0,
                strict_minimum=True,
                error_type=PhysicsValidationError,
            ),
        )


@dataclass(frozen=True)
class SubstepRecommendation:
    """Fixed substep count recommendation for one macro timestep."""

    characteristic_timescale: float
    target_substep_timestep: float
    actual_substep_timestep: float
    substep_count: int
    limited_by_maximum_substeps: bool
    limited_by_minimum_timestep: bool
    would_trigger_refsafe_at_macro_dt: bool
    satisfies_configured_timeconst_at_substep_dt: bool | None


def estimate_solver_contact_timescale(params: MuJoCoContactSolverParams) -> SolverContactTimescale:
    """Estimate the fastest relevant soft-contact timescale from MuJoCo solref/solimp."""
    if not isinstance(params, MuJoCoContactSolverParams):
        raise PhysicsValidationError(f"params must be MuJoCoContactSolverParams; actual value={params!r}")
    d_width = _finite_float(
        params.solimp[1],
        field_name="solimp[1]",
        minimum=0.0,
        strict_minimum=True,
        error_type=PhysicsValidationError,
    )
    assumed_impedance = max(params.solimp[0], params.solimp[1])
    assumed_impedance = _finite_float(
        assumed_impedance,
        field_name="assumed_impedance",
        minimum=0.0,
        strict_minimum=True,
        error_type=PhysicsValidationError,
    )
    first, second = params.solref
    if first > 0.0 and second > 0.0:
        solref_format = "positive"
        timeconst = _finite_float(first, field_name="timeconst", minimum=0.0, strict_minimum=True, error_type=PhysicsValidationError)
        dampratio = _finite_float(second, field_name="dampratio", minimum=0.0, strict_minimum=True, error_type=PhysicsValidationError)
        effective_damping = 2.0 * assumed_impedance / (d_width * timeconst)
        effective_stiffness = (assumed_impedance * assumed_impedance) / (
            d_width * d_width * timeconst * timeconst * dampratio * dampratio
        )
    elif first < 0.0 and second <= 0.0:
        solref_format = "direct"
        stiffness = _finite_float(-first, field_name="direct stiffness", minimum=0.0, strict_minimum=True, error_type=PhysicsValidationError)
        damping = _finite_float(-second, field_name="direct damping", minimum=0.0, error_type=PhysicsValidationError)
        effective_damping = assumed_impedance * damping / d_width
        effective_stiffness = assumed_impedance * assumed_impedance * stiffness / (d_width * d_width)
    else:
        raise PhysicsValidationError(
            "solref must use either positive format (timeconst, dampratio) or direct format (-stiffness, -damping); "
            f"actual value={params.solref!r}"
        )
    return _timescale_from_coefficients(
        solref_format=solref_format,
        assumed_impedance=assumed_impedance,
        impedance_width_value=d_width,
        effective_damping=effective_damping,
        effective_stiffness=effective_stiffness,
    )


def recommend_solver_substeps(
    *,
    macro_timestep: float,
    timescale: SolverContactTimescale,
    params: MuJoCoContactSolverParams,
    config: SubstepRecommendationConfig | None = None,
) -> SubstepRecommendation:
    """Recommend a fixed substep count from an estimated solver contact timescale."""
    macro_dt = _finite_float(
        macro_timestep,
        field_name="macro_timestep",
        minimum=0.0,
        strict_minimum=True,
        error_type=PhysicsValidationError,
    )
    selected_config = config or SubstepRecommendationConfig()
    characteristic = _finite_float(
        timescale.characteristic_timescale,
        field_name="characteristic_timescale",
        minimum=0.0,
        strict_minimum=True,
        error_type=PhysicsValidationError,
    )
    target_dt = characteristic / selected_config.samples_per_characteristic_time
    requested_substeps = max(1, math.ceil(macro_dt / target_dt))
    limited_by_maximum = requested_substeps > selected_config.maximum_substeps
    substep_count = min(requested_substeps, selected_config.maximum_substeps)
    min_dt_max_substeps = math.floor(macro_dt / selected_config.minimum_substep_timestep)
    limited_by_minimum = False
    if min_dt_max_substeps < 1:
        limited_by_minimum = True
        substep_count = 1
    elif substep_count > min_dt_max_substeps:
        limited_by_minimum = True
        substep_count = min_dt_max_substeps
    actual_dt = macro_dt / substep_count
    positive_solref = params.solref[0] > 0.0 and params.solref[1] > 0.0
    would_trigger_refsafe = bool(positive_solref and macro_dt > params.solref[0] / 2.0)
    satisfies_timeconst = None if not positive_solref else actual_dt <= params.solref[0] / 2.0
    return SubstepRecommendation(
        characteristic_timescale=characteristic,
        target_substep_timestep=target_dt,
        actual_substep_timestep=actual_dt,
        substep_count=substep_count,
        limited_by_maximum_substeps=limited_by_maximum,
        limited_by_minimum_timestep=limited_by_minimum,
        would_trigger_refsafe_at_macro_dt=would_trigger_refsafe,
        satisfies_configured_timeconst_at_substep_dt=satisfies_timeconst,
    )


def _timescale_from_coefficients(
    *,
    solref_format: str,
    assumed_impedance: float,
    impedance_width_value: float,
    effective_damping: float,
    effective_stiffness: float,
) -> SolverContactTimescale:
    natural_frequency = math.sqrt(effective_stiffness)
    damping_ratio = effective_damping / (2.0 * natural_frequency)
    discriminant = effective_damping * effective_damping - 4.0 * effective_stiffness
    tolerance = max(effective_damping * effective_damping, effective_stiffness, 1.0) * 1.0e-12
    if discriminant < -tolerance:
        damped_frequency = math.sqrt(effective_stiffness - effective_damping * effective_damping / 4.0)
        oscillatory_duration = math.pi / damped_frequency
        fastest_mode_timescale = 1.0 / natural_frequency
        regime = DampingRegime.UNDERDAMPED
        characteristic = oscillatory_duration
    elif abs(discriminant) <= tolerance:
        damped_frequency = None
        oscillatory_duration = None
        fastest_mode_timescale = 2.0 / effective_damping
        regime = DampingRegime.CRITICAL
        characteristic = fastest_mode_timescale
    else:
        root = math.sqrt(discriminant)
        lambda_1 = (-effective_damping + root) / 2.0
        lambda_2 = (-effective_damping - root) / 2.0
        fastest_mode_timescale = 1.0 / max(abs(lambda_1), abs(lambda_2))
        damped_frequency = None
        oscillatory_duration = None
        regime = DampingRegime.OVERDAMPED
        characteristic = fastest_mode_timescale
    return SolverContactTimescale(
        solref_format=solref_format,
        assumed_impedance=assumed_impedance,
        impedance_width_value=impedance_width_value,
        effective_damping=effective_damping,
        effective_stiffness=effective_stiffness,
        natural_frequency=natural_frequency,
        damping_ratio=damping_ratio,
        regime=regime,
        damped_frequency=damped_frequency,
        oscillatory_contact_duration=oscillatory_duration,
        fastest_mode_timescale=fastest_mode_timescale,
        characteristic_timescale=characteristic,
    )
