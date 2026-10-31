"""Independent analytic pressure-volume work checks signs and baseline removal."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "examples/assembly_work_study.py"
spec = importlib.util.spec_from_file_location("assembly_work_study", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.mark.parametrize("rate", [-0.2, 0.2])
def test_constant_pressure_work_matches_exact_volume_change(rate):
    pressure, initial_volume, initial_energy = 2.0, 8.0, 100.0

    def row(time):
        cell = np.full(3, 2 * np.exp(rate * time))
        work = pressure * (initial_volume - np.prod(cell))
        return dict(
            cell=cell.tolist(),
            stress=(np.eye(3) * pressure).tolist(),
            kinetic_energy=0.0,
            elastic_energy_proxy=initial_energy + work,
            cell_work=5.0 + work,
            friction_dissipation=7.0,
            drag_dissipation=9.0,
            study_time_s=time,
            interval_rate_s=[rate] * 3,
        )

    result = study.summarize(row(0), [row(t) for t in np.linspace(0.01, 1, 100)])
    exact = pressure * initial_volume * (1 - np.exp(3 * rate))
    assert np.sign(result["independent_cell_work_J"]) == np.sign(-rate)
    assert result["independent_cell_work_J"] == pytest.approx(exact, rel=4e-6)
    assert result["maximum_relative_proxy_balance_residual"] < 1e-14
    assert result["minimum_incremental_dissipation_J"] == 0


def test_nonincreasing_time_cannot_be_a_work_audit():
    row = dict(kinetic_energy=1.0, elastic_energy_proxy=0.0, study_time_s=0.0)
    with pytest.raises(ValueError, match="increasing"):
        study.summarize(row, [row])
