import math

import pytest

from physical_simulation.mujoco import (
    DampingRegime,
    MuJoCoContactSolverParams,
    SubstepRecommendationConfig,
    estimate_solver_contact_timescale,
    recommend_solver_substeps,
)
from physical_simulation.validation.errors import PhysicsValidationError


def _params(solref):
    return MuJoCoContactSolverParams(solref=solref, solimp=(0.9, 0.9, 0.001, 0.5, 2.0))


@pytest.mark.parametrize(
    ("solref", "expected_duration"),
    [
        ((0.02, 0.3), 0.01976),
        ((0.02, 0.5), 0.03628),
    ],
)
def test_positive_solref_underdamped_timescale_regression(solref, expected_duration) -> None:
    timescale = estimate_solver_contact_timescale(_params(solref))

    assert timescale.solref_format == "positive"
    assert timescale.assumed_impedance == pytest.approx(0.9)
    assert timescale.impedance_width_value == pytest.approx(0.9)
    assert timescale.regime is DampingRegime.UNDERDAMPED
    assert timescale.oscillatory_contact_duration == pytest.approx(expected_duration, rel=5.0e-4)
    simplified = math.pi * solref[0] * solref[1] / math.sqrt(1.0 - solref[1] * solref[1])
    assert timescale.characteristic_timescale == pytest.approx(simplified)


def test_positive_solref_critical_and_overdamped() -> None:
    critical = estimate_solver_contact_timescale(_params((0.02, 1.0)))
    overdamped = estimate_solver_contact_timescale(_params((0.02, 1.5)))

    assert critical.regime is DampingRegime.CRITICAL
    assert critical.oscillatory_contact_duration is None
    assert critical.characteristic_timescale > 0.0
    assert overdamped.regime is DampingRegime.OVERDAMPED
    assert overdamped.oscillatory_contact_duration is None
    assert overdamped.characteristic_timescale > 0.0


def test_direct_solref_zero_damping_and_damped_cases() -> None:
    zero_damping = estimate_solver_contact_timescale(_params((-10000.0, -0.0)))
    damped = estimate_solver_contact_timescale(_params((-10000.0, -300.0)))

    assert zero_damping.solref_format == "direct"
    assert zero_damping.regime is DampingRegime.UNDERDAMPED
    assert zero_damping.damping_ratio == pytest.approx(0.0)
    assert damped.effective_damping == pytest.approx(300.0)
    assert damped.effective_stiffness == pytest.approx(10000.0)
    assert damped.regime is DampingRegime.OVERDAMPED


def test_timescale_rejects_invalid_solref_and_solimp() -> None:
    with pytest.raises(PhysicsValidationError, match="solref"):
        estimate_solver_contact_timescale(_params((0.02, -0.5)))
    with pytest.raises(PhysicsValidationError, match="solref"):
        estimate_solver_contact_timescale(_params((-0.0, -1.0)))
    with pytest.raises(PhysicsValidationError, match=r"solimp\[1\]"):
        estimate_solver_contact_timescale(
            MuJoCoContactSolverParams(solref=(0.02, 0.5), solimp=(0.9, 0.0, 0.001, 0.5, 2.0))
        )


def test_recommend_solver_substeps_and_limits() -> None:
    params = _params((0.02, 0.5))
    timescale = estimate_solver_contact_timescale(params)

    recommendation = recommend_solver_substeps(
        macro_timestep=1.0 / 240.0,
        timescale=timescale,
        params=params,
        config=SubstepRecommendationConfig(samples_per_characteristic_time=16),
    )
    assert recommendation.substep_count == 2
    assert recommendation.actual_substep_timestep == pytest.approx(1.0 / 480.0)
    assert recommendation.would_trigger_refsafe_at_macro_dt is False
    assert recommendation.satisfies_configured_timeconst_at_substep_dt is True

    max_limited = recommend_solver_substeps(
        macro_timestep=0.1,
        timescale=timescale,
        params=params,
        config=SubstepRecommendationConfig(samples_per_characteristic_time=1000, maximum_substeps=8),
    )
    assert max_limited.substep_count == 8
    assert max_limited.limited_by_maximum_substeps is True

    min_limited = recommend_solver_substeps(
        macro_timestep=0.1,
        timescale=timescale,
        params=params,
        config=SubstepRecommendationConfig(samples_per_characteristic_time=1000, maximum_substeps=128, minimum_substep_timestep=0.02),
    )
    assert min_limited.substep_count == 5
    assert min_limited.limited_by_minimum_timestep is True
    assert min_limited.actual_substep_timestep == pytest.approx(0.02)


def test_recommendation_records_refsafe_diagnostic() -> None:
    params = _params((0.02, 0.5))
    timescale = estimate_solver_contact_timescale(params)

    recommendation = recommend_solver_substeps(
        macro_timestep=0.02,
        timescale=timescale,
        params=params,
        config=SubstepRecommendationConfig(samples_per_characteristic_time=16),
    )

    assert recommendation.would_trigger_refsafe_at_macro_dt is True
    assert recommendation.satisfies_configured_timeconst_at_substep_dt is True
