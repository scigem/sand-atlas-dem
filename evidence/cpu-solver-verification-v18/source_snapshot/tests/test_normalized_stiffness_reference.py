"""Independent force oracle checked against exact flat-face contact algebra."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "examples"))
try:
    spec = importlib.util.spec_from_file_location(
        "normalized_stiffness_reference",
        root / "examples/normalized_stiffness_study.py",
    )
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("fraction", [0.001, 0.01, 0.05])
def test_convex_reference_integrates_exact_opposing_flat_faces(fraction):
    h = 1e-3
    depth = h * fraction
    cube = trimesh.creation.box(extents=[h] * 3)
    shape = SimpleNamespace(
        vertices=cube.vertices,
        faces=cube.faces,
        samples=np.concatenate([np.eye(3), -np.eye(3)]) * h / 2,
        areas=np.full(6, h * h),
    )
    result = study.oracle(shape, np.tile([0, 0, 0, 1.0], (2, 1)), [0, 0], h - depth)
    np.testing.assert_allclose(
        result["force_integral_m3"], [-h * h * depth, 0, 0], rtol=1e-12, atol=1e-25
    )
    np.testing.assert_allclose(result["torque_integral_m4"], 0, atol=1e-25)
    assert result["maximum_depth_m"] == pytest.approx(depth, rel=1e-12)
    assert result["active_contacts"] == 2


def test_separated_convex_reference_is_exactly_empty():
    cube = trimesh.creation.box(extents=[1e-3] * 3)
    shape = SimpleNamespace(
        vertices=cube.vertices,
        faces=cube.faces,
        samples=np.concatenate([np.eye(3), -np.eye(3)]) * 5e-4,
        areas=np.full(6, 1e-6),
    )
    result = study.oracle(shape, np.tile([0, 0, 0, 1.0], (2, 1)), [0, 0], 0.002)
    np.testing.assert_array_equal(result["force_integral_m3"], 0)
    np.testing.assert_array_equal(result["torque_integral_m4"], 0)
    assert result["active_contacts"] == 0
