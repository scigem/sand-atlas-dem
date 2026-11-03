"""Whole-triangle rejection preserves force, depth and all interior samples."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh

path = Path(__file__).resolve().parents[1] / "examples/library_quadrature_culled.py"
spec = importlib.util.spec_from_file_location("culled_library_quadrature", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.mark.parametrize("concave", [False, True])
@pytest.mark.parametrize("subdivisions", [0, 2])
def test_triangle_rejection_preserves_integral(concave, subdivisions):
    mesh = (
        trimesh.creation.annulus(r_min=0.4, r_max=1, height=1, sections=24)
        if concave
        else trimesh.creation.box()
    )
    planes = study.geometry.enclosure_planes(mesh.vertices)
    original = np.random.default_rng(392).uniform(-1.5, 1.5, (120, 3, 3))
    culled = study.cull_source_triangles(original, mesh.bounds, planes)
    for _ in range(subdivisions):
        original = study.geometry.subdivide(original)
        culled = study.cull_source_triangles(
            study.geometry.subdivide(culled), mesh.bounds, planes
        )
    expected = study.geometry.integrate(mesh, original, 1, 64, planes)
    actual = study.integrate(mesh, culled, 1, 64, planes)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-14)
    assert actual[1:3] == expected[1:3]
    assert actual[2] > 0
    assert actual[3] < expected[3]


def test_target_boundary_triangles_are_retained():
    mesh = trimesh.creation.box()
    planes = study.geometry.enclosure_planes(mesh.vertices)
    result = study.cull_source_triangles(mesh.triangles, mesh.bounds, planes)
    assert np.array_equal(result, mesh.triangles)
