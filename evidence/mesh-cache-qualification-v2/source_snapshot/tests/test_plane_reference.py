"""Closed plane-contact integrals checked by independent scalar quadrature."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

path = Path(__file__).resolve().parents[1] / "examples/plane_reference.py"
spec = importlib.util.spec_from_file_location("plane_integral_reference", path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.mark.parametrize("radius", [0.01, 1.0])
@pytest.mark.parametrize("transition_ratio", [0.0, 0.25, 2.0])
def test_closed_force_torque_and_normal_work(radius, transition_ratio):
    depth, kn, kt, mu = 0.2 * radius, 7.0, 3.0, 0.4
    slip = transition_ratio * mu * kn * depth / kt
    result = reference.sphere_plane(radius, depth, kn, kt, mu, slip)
    # Parameterize the original surface by its height z, not the closed form's d.
    lower, upper = -radius, -radius + depth
    kink = upper - kt * slip / (mu * kn)
    points = [kink] if lower < kink < upper else None
    traction = lambda z: min(kt * slip, mu * kn * (upper - z))
    normal = (
        2
        * np.pi
        * radius
        * quad(lambda z: kn * (upper - z), lower, upper, epsabs=1e-20)[0]
    )
    tangent = (
        -2
        * np.pi
        * radius
        * quad(traction, lower, upper, points=points, epsabs=1e-20)[0]
    )
    torque = (
        -2
        * np.pi
        * radius
        * quad(lambda z: z * traction(z), lower, upper, points=points, epsabs=1e-20)[0]
    )
    np.testing.assert_allclose(
        result["force"], [tangent, 0.0, normal], rtol=1e-12, atol=1e-20
    )
    np.testing.assert_allclose(
        result["torque"], [0.0, torque, 0.0], rtol=1e-12, atol=1e-20
    )
    work = quad(
        lambda delta: reference.sphere_plane(radius, delta, kn, kt, mu, 0.0)["force"][
            2
        ],
        0.0,
        depth,
        epsabs=1e-20,
    )[0]
    assert result["normal_energy"] == pytest.approx(work, rel=1e-12)
