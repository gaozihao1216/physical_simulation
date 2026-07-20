"""Mass and diagonal inertia properties for rigid bodies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from physical_simulation.validation.asset_validator import _as_float_tuple, _finite_float
from physical_simulation.validation.errors import InvalidMassPropertiesError


@dataclass(frozen=True)
class MassProperties:
    """Mass, center of mass, and diagonal inertia in SI units."""

    mass: float
    center_of_mass: tuple[float, float, float]
    inertia_diagonal: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mass",
            _finite_float(
                self.mass,
                field_name="mass",
                minimum=0.0,
                strict_minimum=True,
                error_type=InvalidMassPropertiesError,
            ),
        )
        object.__setattr__(
            self,
            "center_of_mass",
            _as_float_tuple(
                self.center_of_mass,
                field_name="center_of_mass",
                length=3,
                error_type=InvalidMassPropertiesError,
            ),
        )
        object.__setattr__(
            self,
            "inertia_diagonal",
            _as_float_tuple(
                self.inertia_diagonal,
                field_name="inertia_diagonal",
                length=3,
                strictly_positive=True,
                error_type=InvalidMassPropertiesError,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize mass properties to a JSON-compatible dictionary."""
        return {
            "mass": self.mass,
            "center_of_mass": list(self.center_of_mass),
            "inertia_diagonal": list(self.inertia_diagonal),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MassProperties":
        """Deserialize mass properties from a dictionary."""
        if not isinstance(data, dict):
            raise InvalidMassPropertiesError(f"mass_properties data must be a dict; actual value={data!r}")
        return cls(
            mass=data.get("mass"),
            center_of_mass=tuple(data.get("center_of_mass", ())),
            inertia_diagonal=tuple(data.get("inertia_diagonal", ())),
        )
