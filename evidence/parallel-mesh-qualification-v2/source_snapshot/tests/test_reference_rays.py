"""Ray pruning must preserve upstream hit sets and parity, including cavities."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh

path = Path(__file__).resolve().parents[1] / "examples/reference_rays.py"
spec = importlib.util.spec_from_file_location("reference_rays", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("concave", [False, True])
def test_padded_ray_box_filter_preserves_upstream_hits_and_parity(concave):
    mesh = (
        trimesh.creation.annulus(r_min=0.4, r_max=1, height=1, sections=32)
        if concave
        else trimesh.creation.box(extents=[1, 2, 3])
    )
    rng = np.random.default_rng(417)
    origins = rng.uniform(-2, 2, (150, 3))
    directions = rng.normal(size=origins.shape)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    # Include axis rays, nearly parallel rays and exact source vertices/edges.
    origins = np.vstack([origins, mesh.vertices[:6], np.zeros((6, 3))])
    directions = np.vstack(
        [
            directions,
            np.eye(3),
            -np.eye(3),
            [
                [1, 1e-14, 0],
                [0, 1, 1e-14],
                [1e-14, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ],
        ]
    )
    accelerated = module.BoundedRayIntersector(mesh)
    expected = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
    actual = accelerated.intersects_location(origins, directions, multiple_hits=True)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    # Surface points have undefined parity in upstream; test interior/exterior
    # random points and explicit cavity points instead.
    points = np.vstack([rng.uniform(-1.5, 1.5, (200, 3)), [[0, 0, 0], [0.7, 0, 0]]])
    np.testing.assert_array_equal(
        accelerated.contains_points(points), mesh.contains(points)
    )
