"""Independent overlap error cases, including zero references and JSON-safe failure."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

source = Path(__file__).resolve().parents[1] / "examples/refine_quadrature_case.py"
spec = importlib.util.spec_from_file_location("refinement_overlap", source)
refinement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refinement)


def test_zero_reference_requires_exact_zero_and_serializes_failure():
    assert refinement.overlap_difference([0, 0, 0], [0, 0, 0]) == 0
    assert refinement.overlap_difference([0, 0, 0], [0, 1e-30, 0]) is None
    assert (
        json.dumps(
            dict(difference=refinement.overlap_difference([0, 0, 0], [1, 0, 0])),
            allow_nan=False,
        )
        == '{"difference": null}'
    )


def test_relative_error_uses_original_nonzero_force_norm():
    assert refinement.overlap_difference(
        [3e-17, 4e-17, 0], [3e-17, 4e-17, 5e-20]
    ) == pytest.approx(0.001)
    assert refinement.overlap_difference([3, 4, 0], [0, 0, 0]) == 1
    assert refinement.overlap_difference([3, 4, 0], [3, 4, 0]) == 0


@pytest.mark.parametrize(
    "a,b",
    [
        ([np.nan, 0, 0], [0, 0, 0]),
        ([0, 0, 0], [np.inf, 0, 0]),
        ([1, 2], [1, 2]),
        ([[1, 2, 3]], [[1, 2, 3]]),
    ],
)
def test_nonfinite_or_wrong_shape_reference_fails(a, b):
    with pytest.raises(ValueError, match="Finite three-vector"):
        refinement.overlap_difference(a, b)
