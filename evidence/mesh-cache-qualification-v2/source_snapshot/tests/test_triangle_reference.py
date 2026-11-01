"""Analytical minima and exhaustive searches qualify the FP64 query oracle."""

import numpy as np
import pytest
import trimesh

from sand_atlas_dem.reference import triangle_nearest_points


def test_nearest_point_keeps_minimum_when_farther_face_has_preferred_normal():
    # The closer triangle points down; the farther one points up. Normal-based
    # tie substitution must not move the nearest point to the farther plane.
    mesh = trimesh.Trimesh(
        vertices=[
            [-1, -1, 0],
            [0, 1, 0],
            [1, -1, 0],
            [-1, -1, -2e-6],
            [1, -1, -2e-6],
            [0, 1, -2e-6],
        ],
        faces=[[0, 1, 2], [3, 4, 5]],
        process=False,
    )
    nearest, distance, face = triangle_nearest_points(mesh, [[0, 0, 0.001]])
    assert np.allclose(nearest, [[0, 0, 0]], rtol=0, atol=1e-15)
    assert distance[0] == pytest.approx(0.001, rel=1e-14)
    assert face[0] == 0


def test_candidate_search_matches_exhaustive_triangle_minimum():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    points = np.random.default_rng(517).uniform(-1.5, 1.5, (80, 3))
    nearest, distance, _ = triangle_nearest_points(mesh, points)
    for point, actual, length in zip(points, nearest, distance):
        candidates = trimesh.triangles.closest_point(
            mesh.triangles, np.broadcast_to(point, mesh.triangles.shape[:1] + (3,))
        )
        squared = np.sum((candidates - point) ** 2, axis=1)
        assert length**2 == pytest.approx(squared.min(), abs=1e-14)
        assert np.sum((actual - point) ** 2) == pytest.approx(squared.min(), abs=1e-14)


def test_nearest_query_empty_and_invalid_inputs():
    mesh = trimesh.creation.box()
    assert triangle_nearest_points(mesh, np.empty((0, 3)))[0].shape == (0, 3)
    with pytest.raises(ValueError, match="finite"):
        triangle_nearest_points(mesh, [[np.nan, 0, 0]])
