"""Independent quadrature and parameter guards for the damping scaling audit."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

_spec = importlib.util.spec_from_file_location(
    "restitution_audit",
    Path(__file__).parents[1] / "examples/restitution_scaling_audit.py",
)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


@pytest.mark.parametrize("depth", [0.00001, 0.00004, 0.00016])
def test_cap_damping_matches_independent_surface_quadrature(depth):
    radius, stiffness, viscosity, speed = 0.01, 1e12, 1e5, 0.001
    value = audit.sphere_cap(radius, depth, stiffness, viscosity, speed)
    # Integrate along z on the original sphere, with dA=2*pi*R*dz.
    offset = -radius + depth
    elastic = quad(
        lambda z: stiffness * (offset - z) * 2 * np.pi * radius, -radius, offset
    )[0]
    damping = quad(lambda z: viscosity * speed * 2 * np.pi * radius, -radius, offset)[0]
    power = quad(lambda z: viscosity * speed**2 * 2 * np.pi * radius, -radius, offset)[
        0
    ]
    assert np.isclose(value["elastic_force"], elastic, rtol=1e-11, atol=0)
    assert np.isclose(value["damping_force"], damping, rtol=1e-11, atol=0)
    assert np.isclose(value["dissipation_power"], power, rtol=1e-11, atol=0)


@pytest.mark.parametrize("restitution", [-1, 0, 1.01, float("nan")])
def test_invalid_restitution_is_not_clamped(restitution):
    with pytest.raises(ValueError):
        audit.damping_ratio(restitution)


def test_ordinary_oscillator_inverse_and_traction_units_are_distinct():
    for restitution in [0.01, 0.1, 0.5, 0.9, 1]:
        assert np.isclose(
            audit.linear_restitution(audit.damping_ratio(restitution)),
            restitution,
            rtol=1e-12,
        )
    # Both total K and C must be invariant when one patch is split into 16 nodes.
    a = audit.patch_coefficients(1, 0.2, 12, 5, 0.5)
    b = audit.patch_coefficients(16, 0.2, 12, 5, 0.5)
    assert np.isclose(b["area_partitioned_mass"], a["area_partitioned_mass"])
    assert np.isclose(
        b["full_body_mass_at_every_node"], 4 * a["full_body_mass_at_every_node"]
    )
