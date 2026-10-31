"""Reordering parity and distance queries must preserve the strict integral."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh

path = Path(__file__).resolve().parents[1] / "examples/library_quadrature.py"
spec = importlib.util.spec_from_file_location("library_quadrature", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.mark.parametrize("concave", [False, True])
@pytest.mark.parametrize("enclosure", [False, True])
def test_inside_first_preserves_force_depth_and_active_count(concave, enclosure):
    mesh = (
        trimesh.creation.annulus(r_min=0.4, r_max=1, height=1, sections=24)
        if concave
        else trimesh.creation.box(extents=[1, 2, 3])
    )
    triangles = np.random.default_rng(319).uniform(-1.5, 1.5, (120, 3, 3))
    planes = study.geometry.enclosure_planes(mesh.vertices) if enclosure else None
    expected = study.geometry.integrate(mesh, triangles, 1, 64, planes)
    actual = study.integrate(mesh, triangles, 1, 64, planes)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-14)
    assert actual[1:4] == expected[1:4]
    assert actual[2] > 0
    assert actual[4] == actual[2] <= expected[4]
    if concave or not enclosure:
        assert actual[4] < expected[4]
    # A box's exact convex enclosure already removes every exterior candidate;
    # query reordering preserves its count as well as its integral.
