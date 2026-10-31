"""Deterministic inside classification preserves analytical solids and cavities."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh

path = Path(__file__).resolve().parents[1] / "examples/reference_inside.py"
spec = importlib.util.spec_from_file_location("reference_inside", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("concave", [False, True])
def test_inside_is_independent_of_query_order_and_batch_size(concave):
    mesh = (
        trimesh.creation.annulus(r_min=0.4, r_max=1, height=1, sections=32)
        if concave
        else trimesh.creation.box(extents=[1, 2, 3])
    )
    query = module.DeterministicRayIntersector(mesh)
    p = np.random.default_rng(418).uniform(-1.5, 1.5, (170, 3))
    expected = mesh.contains(p)
    actual = query.contains_points(p)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(query.contains_points(p[::-1])[::-1], actual)
    split = np.concatenate(
        [query.contains_points(p[i : i + 7]) for i in range(0, len(p), 7)]
    )
    np.testing.assert_array_equal(split, actual)
    if concave:
        np.testing.assert_array_equal(
            query.contains_points([[0, 0, 0], [0.7, 0, 0]]), [False, True]
        )
