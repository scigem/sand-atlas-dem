"""Study continuation must preserve physical input and all acceptance gates."""

import importlib.util
from pathlib import Path

import pytest

from sand_atlas_dem.spec import Experiment

_source = Path(__file__).resolve().parents[1] / "examples/controller_study.py"
_module_spec = importlib.util.spec_from_file_location("controller_study", _source)
_driver = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(_driver)
reconcile = _driver.reconcile_resume


def test_resume_preserves_effective_allocation_without_changing_protocol():
    existing = Experiment(contacts_per_particle=512)
    requested = Experiment(contacts_per_particle=256)
    old = existing.to_dict()
    new = requested.to_dict()
    result = reconcile(existing, requested)
    assert result.contacts_per_particle == 512
    assert existing.to_dict() == old
    assert requested.to_dict() == new


def test_budget_extension_is_explicit_and_cannot_shorten():
    existing = Experiment()
    requested = Experiment(stages=[{"kind": "hold", "steps": 2000, "drag": 0.0}])
    with pytest.raises(ValueError, match="physical protocol"):
        reconcile(existing, requested)
    assert (
        reconcile(existing, requested, extend_step_budgets=True).stages[0]["steps"]
        == 2000
    )
    with pytest.raises(ValueError, match="shorten"):
        reconcile(requested, existing, extend_step_budgets=True)


@pytest.mark.parametrize("field,value", [("friction", 0.1), ("timestep", 1e-8)])
def test_budget_extension_does_not_authorize_physical_changes(field, value):
    existing = Experiment()
    requested = Experiment(**{field: value})
    with pytest.raises(ValueError, match="physical protocol"):
        reconcile(existing, requested, extend_step_budgets=True)


def test_budget_extension_does_not_authorize_relaxed_acceptance():
    stage = {
        "kind": "isotropic",
        "steps": 1000,
        "drag": 0,
        "pressure": 10000,
        "gain": 1000,
        "filter_time": 1e-5,
        "max_rate": 50,
        "kinetic_tolerance": 1e-5,
    }
    existing = Experiment(stages=[stage])
    requested = Experiment(stages=[dict(stage, steps=2000, kinetic_tolerance=1e-4)])
    with pytest.raises(ValueError, match="physical protocol"):
        reconcile(existing, requested, extend_step_budgets=True)
