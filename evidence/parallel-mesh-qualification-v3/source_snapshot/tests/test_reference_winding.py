"""Closed solids, cavities and orientation independently check solid angles."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh

path = Path(__file__).resolve().parents[1] / "examples/reference_winding.py"
spec = importlib.util.spec_from_file_location("reference_winding", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_winding_number_has_analytical_box_inside_outside_and_orientation():
    mesh = trimesh.creation.box(extents=[1, 2, 3])
    points = [[0, 0, 0], [0.49, 0.8, 1.4], [0.51, 0, 0], [0, 1.1, 0], [0, 0, 2]]
    for point, expected in zip(points, [1, 1, 0, 0, 0]):
        assert module.winding_number(mesh.triangles, point, batch=5) == pytest.approx(
            expected, abs=1e-12
        )
        assert module.winding_number(mesh.triangles[:, ::-1], point) == pytest.approx(
            -expected, abs=1e-12
        )


def test_winding_number_distinguishes_concave_solid_from_its_cavity():
    mesh = trimesh.creation.annulus(r_min=0.4, r_max=1, height=1, sections=32)
    for point, expected in zip(
        [[0, 0, 0], [0.7, 0, 0], [1.1, 0, 0], [0.7, 0, 0.6]], [0, 1, 0, 0]
    ):
        assert module.winding_number(mesh.triangles, point) == pytest.approx(
            expected, abs=1e-12
        )
    with pytest.raises(ValueError, match="undefined"):
        module.winding_number(mesh.triangles, mesh.vertices[0])
