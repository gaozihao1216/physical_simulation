import pytest

from physical_simulation.mujoco import (
    AnalyticPlane,
    DampingRegime,
    MuJoCoContactSolverParams,
    estimate_solver_collision,
    predict_sphere_plane_collision,
    predict_sphere_sphere_collision,
)


def test_sphere_plane_predicts_approaching_contact() -> None:
    prediction = predict_sphere_plane_collision(
        sphere_position=(0.0, 0.0, 1.0),
        sphere_velocity=(0.0, 0.0, -2.0),
        sphere_radius=0.1,
        plane=AnalyticPlane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 2.0)),
        prediction_horizon=1.0,
    )

    assert prediction is not None
    assert prediction.collision_type == "sphere_plane"
    assert prediction.gap == pytest.approx(0.9)
    assert prediction.time_to_contact == pytest.approx(0.45)
    assert prediction.normal_approach_speed == pytest.approx(2.0)
    assert prediction.contact_normal == pytest.approx((0.0, 0.0, 1.0))


def test_sphere_plane_ignores_receding_and_beyond_horizon() -> None:
    plane = AnalyticPlane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))

    assert predict_sphere_plane_collision(
        sphere_position=(0.0, 0.0, 1.0),
        sphere_velocity=(0.0, 0.0, 1.0),
        sphere_radius=0.1,
        plane=plane,
        prediction_horizon=1.0,
    ) is None
    assert predict_sphere_plane_collision(
        sphere_position=(0.0, 0.0, 1.0),
        sphere_velocity=(0.0, 0.0, -0.5),
        sphere_radius=0.1,
        plane=plane,
        prediction_horizon=1.0,
    ) is None


def test_sphere_sphere_head_on_static_and_symmetry() -> None:
    first = predict_sphere_sphere_collision(
        sphere_a_position=(-1.0, 0.0, 0.0),
        sphere_a_velocity=(1.0, 0.0, 0.0),
        sphere_a_radius=0.25,
        sphere_b_position=(1.0, 0.0, 0.0),
        sphere_b_velocity=(0.0, 0.0, 0.0),
        sphere_b_radius=0.25,
        prediction_horizon=3.0,
    )
    second = predict_sphere_sphere_collision(
        sphere_a_position=(1.0, 0.0, 0.0),
        sphere_a_velocity=(0.0, 0.0, 0.0),
        sphere_a_radius=0.25,
        sphere_b_position=(-1.0, 0.0, 0.0),
        sphere_b_velocity=(1.0, 0.0, 0.0),
        sphere_b_radius=0.25,
        prediction_horizon=3.0,
    )

    assert first is not None
    assert second is not None
    assert first.time_to_contact == pytest.approx(1.5)
    assert second.time_to_contact == pytest.approx(first.time_to_contact)
    assert first.normal_approach_speed == pytest.approx(second.normal_approach_speed)


def test_sphere_sphere_ignores_same_velocity_receding_and_miss() -> None:
    assert predict_sphere_sphere_collision(
        sphere_a_position=(0.0, 0.0, 0.0),
        sphere_a_velocity=(1.0, 0.0, 0.0),
        sphere_a_radius=0.25,
        sphere_b_position=(1.0, 0.0, 0.0),
        sphere_b_velocity=(1.0, 0.0, 0.0),
        sphere_b_radius=0.25,
        prediction_horizon=1.0,
    ) is None
    assert predict_sphere_sphere_collision(
        sphere_a_position=(0.0, 0.0, 0.0),
        sphere_a_velocity=(-1.0, 0.0, 0.0),
        sphere_a_radius=0.25,
        sphere_b_position=(1.0, 0.0, 0.0),
        sphere_b_velocity=(1.0, 0.0, 0.0),
        sphere_b_radius=0.25,
        prediction_horizon=1.0,
    ) is None
    assert predict_sphere_sphere_collision(
        sphere_a_position=(0.0, 0.0, 0.0),
        sphere_a_velocity=(1.0, 0.0, 0.0),
        sphere_a_radius=0.25,
        sphere_b_position=(1.0, 2.0, 0.0),
        sphere_b_velocity=(0.0, 0.0, 0.0),
        sphere_b_radius=0.25,
        prediction_horizon=2.0,
    ) is None


def test_sphere_sphere_tangent_and_initial_overlap() -> None:
    tangent = predict_sphere_sphere_collision(
        sphere_a_position=(0.0, 0.0, 0.0),
        sphere_a_velocity=(1.0, 0.0, 0.0),
        sphere_a_radius=0.5,
        sphere_b_position=(1.0, 1.0, 0.0),
        sphere_b_velocity=(0.0, 0.0, 0.0),
        sphere_b_radius=0.5,
        prediction_horizon=2.0,
    )
    overlap = predict_sphere_sphere_collision(
        sphere_a_position=(0.0, 0.0, 0.0),
        sphere_a_velocity=(0.0, 0.0, 0.0),
        sphere_a_radius=0.75,
        sphere_b_position=(1.0, 0.0, 0.0),
        sphere_b_velocity=(0.0, 0.0, 0.0),
        sphere_b_radius=0.75,
        prediction_horizon=1.0,
    )

    assert tangent is not None
    assert tangent.time_to_contact == pytest.approx(1.0)
    assert overlap is not None
    assert overlap.time_to_contact == 0.0
    assert overlap.gap < 0.0


def test_solver_collision_estimate_combines_prediction_and_recommendation() -> None:
    params = MuJoCoContactSolverParams(solref=(0.02, 0.3), solimp=(0.9, 0.9, 0.001, 0.5, 2.0))
    prediction = predict_sphere_plane_collision(
        sphere_position=(0.0, 0.0, 0.2),
        sphere_velocity=(0.0, 0.0, -1.0),
        sphere_radius=0.1,
        plane=AnalyticPlane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)),
        prediction_horizon=1.0,
    )

    estimate = estimate_solver_collision(
        prediction=prediction,
        params=params,
        macro_timestep=1.0 / 240.0,
    )

    assert estimate.prediction is prediction
    assert estimate.timescale.regime is DampingRegime.UNDERDAMPED
    assert estimate.recommendation.substep_count >= 1
