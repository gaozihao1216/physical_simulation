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
    compute_cone_inertia,
    compute_cylinder_inertia,
    compute_ellipsoid_inertia,
    compute_inertia,
    compute_mass_from_density,
    compute_sphere_inertia,
)
from physical_simulation.dynamics.polyhedral_inertia import (
    compute_frustum_mass_properties,
    compute_full_geometry_mass_properties,
    compute_polyhedral_mass_properties,
    compute_regular_prism_mass_properties,
    compute_wedge_mass_properties,
)

__all__ = [
    "CompoundInertiaComponent",
    "CompoundMassProperties",
    "compute_box_inertia",
    "compute_sphere_inertia",
    "compute_cylinder_inertia",
    "compute_capsule_inertia",
    "compute_cone_inertia",
    "compute_ellipsoid_inertia",
    "compute_inertia",
    "compute_mass_from_density",
    "compute_polyhedral_mass_properties",
    "compute_wedge_mass_properties",
    "compute_frustum_mass_properties",
    "compute_regular_prism_mass_properties",
    "compute_full_geometry_mass_properties",
    "compute_compound_mass_properties",
    "diagonalize_symmetric_tensor",
    "rotate_inertia_tensor",
    "translate_inertia_tensor",
]
