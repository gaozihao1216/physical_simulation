"""Rigid-body dynamics configuration and inertia helpers."""

from physical_simulation.dynamics.compound_inertia import (
    CompoundInertiaComponent,
    CompoundMassProperties,
    compute_compound_mass_properties,
    diagonalize_symmetric_tensor,
    rotate_inertia_tensor,
    translate_inertia_tensor,
)
from physical_simulation.dynamics.inertia import (
    compute_box_inertia,
    compute_capsule_inertia,
    compute_cylinder_inertia,
    compute_inertia,
    compute_mass_from_density,
    compute_sphere_inertia,
)

__all__ = [
    "CompoundInertiaComponent",
    "CompoundMassProperties",
    "compute_box_inertia",
    "compute_sphere_inertia",
    "compute_cylinder_inertia",
    "compute_capsule_inertia",
    "compute_inertia",
    "compute_mass_from_density",
    "compute_compound_mass_properties",
    "diagonalize_symmetric_tensor",
    "rotate_inertia_tensor",
    "translate_inertia_tensor",
]
