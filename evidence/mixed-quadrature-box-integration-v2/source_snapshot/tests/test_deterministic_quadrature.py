"""Check the deterministic integral against exhaustive solid-angle labels."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh


def load(name, filename):
    path = Path(__file__).resolve().parents[1] / "examples" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = load("deterministic_quadrature", "library_quadrature_deterministic.py")
winding = load("quadrature_winding", "reference_winding.py")


@pytest.mark.parametrize("concave", [False, True])
def test_integral_matches_exhaustive_winding_and_survives_culling(concave):
    mesh = (
        trimesh.creation.annulus(r_min=0.4, r_max=1, height=1, sections=24)
        if concave
        else trimesh.creation.box()
    )
    mesh.ray = study.rays.DeterministicRayIntersector(mesh)
    triangles = np.random.default_rng(502).uniform(-1.5, 1.5, (80, 3, 3))
    bary = np.full((3, 3), 1 / 6) + np.eye(3) / 2
    points = np.einsum("ij,fjk->fik", bary, triangles).reshape(-1, 3)
    areas = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        / 2
    )
    labels = np.array(
        [abs(winding.winding_number(mesh.triangles, p) - 1) < 1e-8 for p in points]
    )
    nearest, distance, _ = study.triangle_nearest_points(mesh, points[labels])
    expected = 0.5 * np.sum(
        np.repeat(areas / 3, 3)[labels, None] * (nearest - points[labels]), axis=0
    )
    planes = study.geometry.enclosure_planes(mesh.vertices)
    culled = study.cull_source_triangles(triangles, mesh.bounds, planes)
    for source, batch in ((triangles, 64), (triangles[::-1], 7), (culled, 23)):
        force, depth, active, _, queries = study.integrate(
            mesh, source, 1, batch, planes
        )
        np.testing.assert_allclose(force, expected, rtol=1e-12, atol=1e-14)
        assert depth == float(distance.max())
        assert active == queries == int(labels.sum())
