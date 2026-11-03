"""Directional stress correction, overlap safety and old mean-control contract."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('packing_pressure_control', Path(__file__).parents[1]/'examples/packing_pressure_control.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def protocol(mode='diagonal'):
    return dict(pressure_control=mode, target_pressure_Pa=10000., servo_gain=5000.,
                max_rate_s=5000., taper_overlap=.02, hold_overlap=.04, max_batch_log_strain=.0002)


def test_corrects_deviatoric_stress_when_mean_already_on_target():
    stress = np.diag([11000., 9500., 9500.])
    rates, _ = controller.pressure_rates(stress, .03, protocol(), 1e-7)
    assert rates[0] > 0 and np.all(rates[1:] < 0)
    old, _ = controller.pressure_rates(stress, .03, protocol('mean'), 1e-7)
    np.testing.assert_array_equal(old, np.zeros(3))
    perm = [2, 0, 1]
    changed, _ = controller.pressure_rates(stress[np.ix_(perm, perm)], .03, protocol(), 1e-7)
    np.testing.assert_array_equal(changed, rates[perm])


def test_hold_blocks_compression_but_preserves_pressure_relief():
    rates, taper = controller.pressure_rates(np.diag([11000., 9500., 9500.]), .04, protocol(), 1e-7)
    assert rates[0] > 0 and np.all(rates[1:] == 0) and taper == 0


def test_per_axis_strain_budget_and_rate_budget():
    p = protocol()
    stress = np.diag([1e9, 0., 10000.])
    rates, _ = controller.pressure_rates(stress, 0., p, 1e-4)
    assert np.all(abs(rates)*1e-4 <= p['max_batch_log_strain'])
    rates, _ = controller.pressure_rates(stress, 0., p, 1e-9)
    assert np.all(abs(rates) <= p['max_rate_s'])


@pytest.mark.parametrize('mode', ['mean', 'diagonal'])
def test_isotropic_stress_matches_between_modes(mode):
    a, _ = controller.pressure_rates(np.eye(3)*9000, .025, protocol(mode), 1e-7)
    b, _ = controller.pressure_rates(np.eye(3)*9000, .025, protocol('mean'), 1e-7)
    np.testing.assert_array_equal(a, b)
