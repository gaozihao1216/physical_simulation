"""XML formatting helpers."""

from __future__ import annotations

import math
from xml.etree.ElementTree import Element

from physical_simulation.validation.errors import PhysicsValidationError


def format_float(value: float) -> str:
    """Format a finite float using stable, compact round-trip text."""
    number = float(value)
    if not math.isfinite(number):
        raise PhysicsValidationError(f"XML float value must be finite; actual value={value!r}")
    if number == 0.0:
        return "0"
    nearest_integer = round(number)
    if abs(number - nearest_integer) <= 1.0e-12:
        return f"{nearest_integer}.0"
    return repr(number)


def format_vector(values: tuple[float, ...]) -> str:
    """Format a vector as space-separated XML attribute text."""
    return " ".join(format_float(value) for value in values)


def indent_xml(element: Element, level: int = 0) -> None:
    """Indent an ElementTree element in place for readable XML output."""
    indent = "\n" + level * "  "
    child_indent = "\n" + (level + 1) * "  "
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = child_indent
        for child in children:
            indent_xml(child, level + 1)
        if not children[-1].tail or not children[-1].tail.strip():
            children[-1].tail = indent
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent
