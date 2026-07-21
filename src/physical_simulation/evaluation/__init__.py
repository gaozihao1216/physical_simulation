"""Simulation evaluation components."""

from physical_simulation.evaluation.metrics import quaternion_angular_distance, vector_norm
from physical_simulation.evaluation.contact_calibration import (
    ReferenceRestitutionTarget,
    RestitutionMeasurement,
    measure_restitution,
)
from physical_simulation.evaluation.resting_contact import (
    RestingContactMetrics,
    SettlingCriteria,
    evaluate_resting_contact,
)
from physical_simulation.evaluation.trajectory import BodyStateSample, simulate_body_trajectory

__all__ = [
    "BodyStateSample",
    "ReferenceRestitutionTarget",
    "RestitutionMeasurement",
    "RestingContactMetrics",
    "SettlingCriteria",
    "evaluate_resting_contact",
    "measure_restitution",
    "quaternion_angular_distance",
    "simulate_body_trajectory",
    "vector_norm",
]
