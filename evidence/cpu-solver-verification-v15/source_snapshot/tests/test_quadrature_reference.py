"""Outside-point rejection must leave the independent triangle integral intact."""

import importlib.util
from pathlib import Path

import numpy as np
import trimesh


def reference_module():
    path = Path(__file__).parents[1] / "examples/triangle_quadrature_culled.py"
    spec = importlib.util.spec_from_file_location("culled_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_enclosure_keeps_all_analytical_box_interior_and_original_vertices():
    module = reference_module()
    mesh = trimesh.creation.box(extents=[1, 2, 3])
    planes = module.enclosure_planes(mesh.vertices)
    rng = np.random.default_rng(915)
    inside = rng.uniform(-1, 1, (512, 3)) * [0.5, 1, 1.5]
    assert module.enclosure_mask(inside, planes).all()
    assert module.enclosure_mask(mesh.vertices, planes).all()
    assert not module.enclosure_mask(np.array([[0.6, 0, 0], [0, 1.1, 0]]), planes).any()


def test_rejection_preserves_exact_triangle_force_depth_and_active_count():
    module = reference_module()
    mesh = trimesh.creation.box()
    triangles = np.random.default_rng(319).uniform(-2, 2, (100, 3, 3))
    plain = module.integrate(mesh, triangles, 1, 64)
    accelerated = module.integrate(
        mesh, triangles, 1, 64, module.enclosure_planes(mesh.vertices)
    )
    np.testing.assert_allclose(plain[0], accelerated[0], rtol=1e-12, atol=1e-14)
    assert plain[1:4] == accelerated[1:4]
    assert accelerated[2] > 0
    assert accelerated[4] < plain[4]
