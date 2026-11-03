"""A vanishing reference cannot hide an unmatched GPU contact set."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "examples/mesh_quadrature_study.py"
spec = importlib.util.spec_from_file_location("gpu_quadrature_study", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
policy = {"zero_fixed_reference": "require-exact-zero-force-and-zero-active-nodes"}


def test_empty_force_and_contacts_require_explicit_policy():
    zero = np.zeros(3)
    assert study.fixed_level_error(zero, zero, 0, 0, policy) == 0
    with pytest.raises(ValueError, match="explicit policy"):
        study.fixed_level_error(zero, zero, 0, 0, {})


@pytest.mark.parametrize(
    "force,gpu,reference", [([1e-30, 0, 0], 0, 0), ([0, 0, 0], 1, 0), ([0, 0, 0], 0, 2)]
)
def test_zero_force_does_not_excuse_unmatched_contacts(force, gpu, reference):
    with pytest.raises(ValueError, match="empty GPU contact"):
        study.fixed_level_error(np.array(force), np.zeros(3), gpu, reference, policy)


def test_nonzero_reference_keeps_relative_metric_and_rejects_nonfinite():
    assert (
        study.fixed_level_error(np.array([2.0, 0, 0]), np.array([1.0, 0, 0]), 1, 1, {})
        == 1
    )
    with pytest.raises(ValueError, match="Nonfinite"):
        study.fixed_level_error(np.array([np.nan, 0, 0]), np.ones(3), 1, 1, policy)
