"""Pilot cannot bypass preparation or substitute axial for radial control."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def pilot(monkeypatch):
    examples=Path(__file__).parents[1]/'examples'
    monkeypatch.syspath_prepend(str(examples))
    spec=importlib.util.spec_from_file_location('hostun_pilot_contract',examples/'hostun_triaxial_pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def policy():
    return dict(converged_samples=100,target_pressure_Pa=10000,pressure_tolerance=.01,
                kinetic_tolerance=1e-5,force_tolerance=.001,equilibrium_overlap_limit=.05,
                pressure_control='diagonal',servo_gain=5000,max_rate_s=5000,
                taper_overlap=.02,hold_overlap=.04,max_batch_log_strain=.0002)


def test_requires_complete_accepted_preparation_window(pilot):
    report=dict(passed=True,failure=None,source_integrity=True,input_integrity=True)
    row=dict(stress=(np.eye(3)*10000).tolist(),kinetic_energy=1e-8,elastic_energy_proxy=1,
             unbalanced_force=.0005,max_overlap_fraction=.035)
    assert pilot.preparation_ready(report,[row]*100,policy())
    assert not pilot.preparation_ready(report,[row]*99,policy())
    assert not pilot.preparation_ready(dict(report,passed=False),[row]*100,policy())
    bad=dict(row,unbalanced_force=.01)
    assert not pilot.preparation_ready(report,[bad]+[row]*99,policy())


def test_mean_pressure_cannot_hide_directional_failure(pilot):
    report=dict(passed=True,failure=None,source_integrity=True,input_integrity=True)
    row=dict(stress=np.diag([11000,9500,9500]).tolist(),kinetic_energy=0,
             elastic_energy_proxy=1,unbalanced_force=0,max_overlap_fraction=.035)
    assert not pilot.preparation_ready(report,[row]*100,policy())


def test_loading_controls_radial_stresses_and_prescribes_axial_rate(pilot):
    row=dict(stress=np.diag([11000,9500,8000]).tolist(),max_overlap_fraction=.03)
    rates=pilot.loading_rates(row,policy(),1e-8,10,2.)
    assert rates[0]>0 and rates[1]<0 and rates[2]==-2.
