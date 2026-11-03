"""Independent surface integration verifies the deep-overlap counterexample."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'deep_overlap_reproducer', Path(__file__).resolve().parents[1]/'examples/reproduce_deep_overlap.py'
)
reproducer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproducer)


@pytest.mark.parametrize('distance', [.25, .5, .8, 1.2, 1.5, 1.9])
def test_closed_form_matches_independent_surface_traction_integral(distance):
    # R=kn=1. Integrate directly over the penetrating cap's cos(polar angle).
    nodes, weights = np.polynomial.legendre.leggauss(128)
    lower = distance/2
    mu = lower+(nodes+1)*(1-lower)/2
    weights = weights*(1-lower)/2
    radial_distance = np.sqrt(1+distance**2-2*distance*mu)
    depth = 1-radial_distance
    energy = np.pi*np.sum(weights*depth**2)
    force = -2*np.pi*np.sum(weights*depth*(mu-distance)/radial_distance)
    expected_energy, expected_force = reproducer.continuum(distance)
    assert energy/np.pi == pytest.approx(expected_energy, abs=1e-12)
    assert force/np.pi == pytest.approx(expected_force, abs=1e-12)
    assert (force > 0) == (distance > 8/9)


def test_full_coincidence_is_zero_energy_despite_complete_volume_overlap():
    # This is the defect, not a physically acceptable contact requirement.
    assert reproducer.continuum(0) == (0., 0.)
    assert reproducer.continuum(.5)[0] > 0
    assert reproducer.continuum(.5)[1] < 0
