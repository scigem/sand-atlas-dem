"""Boundary conversions for the pinned OpenSees ManzariDafalias 3D wrapper.

Atlas: compression-positive Mandel, Pa. OpenSees: tension-positive stress,
engineering strain, numeric pressure units specified by ``pressure_unit_pa``.
These conventions are material-specific; they are not a universal OpenSees API.
"""
import math

_S = (1., 1., 1., math.sqrt(2.), math.sqrt(2.), math.sqrt(2.))


def _vector(values):
    result = tuple(float(v) for v in values)
    if len(result) != 6 or not all(math.isfinite(v) for v in result):
        raise ValueError('Expected six finite tensor components')
    return result


def _unit(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('pressure_unit_pa must be finite and positive')
    return value


def internal_tensor_to_mandel(values):
    """Stress-like internal tensor, with no sign or unit conversion."""
    return tuple(s*v for s, v in zip(_S, _vector(values)))


def mandel_to_internal_tensor(values):
    return tuple(v/s for s, v in zip(_S, _vector(values)))


def stress_from_opensees(values, *, pressure_unit_pa):
    unit = _unit(pressure_unit_pa)
    return tuple(-unit*v for v in internal_tensor_to_mandel(values))


def stress_to_opensees(values, *, pressure_unit_pa):
    unit = _unit(pressure_unit_pa)
    return tuple(-v/unit for v in mandel_to_internal_tensor(values))


def strain_from_opensees(values):
    return tuple(-v/s for s, v in zip(_S, _vector(values)))


def strain_to_opensees(values):
    return tuple(-s*v for s, v in zip(_S, _vector(values)))


def _matrix(values):
    result = tuple(_vector(row) for row in values)
    if len(result) != 6:
        raise ValueError('Expected a finite 6 by 6 tangent')
    return result


def tangent_from_opensees(values, *, pressure_unit_pa):
    """Map d(stress_external)/d(engineering_strain_external) to Mandel."""
    unit = _unit(pressure_unit_pa)
    return tuple(tuple(unit*_S[i]*v*_S[j] for j, v in enumerate(row))
                 for i, row in enumerate(_matrix(values)))


def tangent_to_opensees(values, *, pressure_unit_pa):
    unit = _unit(pressure_unit_pa)
    return tuple(tuple(v/(_S[i]*_S[j]*unit) for j, v in enumerate(row))
                 for i, row in enumerate(_matrix(values)))
