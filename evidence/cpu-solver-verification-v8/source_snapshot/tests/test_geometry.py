import numpy as np
import pytest
import trimesh

from sand_atlas_dem.geometry import ShapeLibrary, from_mesh, sphere
from sand_atlas_dem.reference import nodal_force, pair_contact, transport_slip
from sand_atlas_dem.spec import Experiment


def test_analytic_sphere_and_roundtrip(tmp_path):
    s = sphere(resolution=32, samples=2048)
    p = tmp_path / "shapes.h5"
    ShapeLibrary([s]).save(p)
    t = ShapeLibrary.load(p).shapes[0]
    points = np.array([[0, 0, 0], [0.0005, 0, 0], [0.0006, 0, 0]])
    value, gradient = t.query(points)
    assert value[0] < 0 and abs(value[1]) < 2e-6 and value[2] > 0
    assert np.allclose(gradient[1], [1, 0, 0], atol=0.04)
    assert np.isclose(t.areas.sum(), 4 * np.pi * 0.0005**2)
    assert np.allclose(t.inertia, np.eye(3) * 0.4 * t.volume * 0.0005**2)


def test_scaled_mesh_and_invalid_geometry():
    mesh = trimesh.creation.box(extents=[1, 2, 3])
    s = from_mesh(mesh, unit="mm", resolution=12, samples=128)
    assert np.isclose(s.volume, 6e-9)
    assert s.query([[0, 0, 0]])[0][0] < 0
    # Signed box distance is analytic at every grid node. The interpolant at
    # the medial axis has O(dx) error because that axis is a distance cusp.
    xyz = np.stack(np.meshgrid(*[np.arange(d) for d in s.sdf.shape], indexing="ij"), -1)
    xyz = s.origin + xyz * s.spacing
    q = np.abs(xyz) - np.array([0.0005, 0.001, 0.0015])
    exact = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)
    assert np.allclose(s.sdf, exact, atol=1e-10, rtol=1e-6)
    assert abs(s.query([[0, 0, 0]])[0][0] + 0.0005) <= s.spacing / 2
    assert abs(s.query([[0, 0, 0.0015]])[0][0]) < s.spacing * 0.05
    bad = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[:-1], process=False)
    with pytest.raises(ValueError):
        from_mesh(bad, unit="mm")
    with pytest.raises(ValueError):
        from_mesh(mesh, unit="guess")


def test_coulomb_return_and_frame_transport():
    f, slip = nodal_force(
        0.1, np.array([0, 0, 1]), 2, np.zeros(3), np.array([1, 0, 0]), 10, 10, 0.3, 1
    )
    assert np.isclose(f[0], -0.3) and np.isclose(f[2], 1)
    assert np.isclose(slip[0], 0.03)
    out = transport_slip([1, 0, 0], [0, 0, 1], [1, 0, 0])
    assert np.allclose(out, [0, 0, -1])
    assert np.allclose(
        transport_slip([1, 0, 0], [0, 0, 1], [0, 0, 1], np.pi / 2), [0, 1, 0]
    )


def test_label_invariance_and_moment_balance():
    s = sphere(resolution=24, samples=1024)
    a = np.array([0.0, 0, 0])
    b = np.array([0.00098, 0, 0])
    r = pair_contact(s, s, a, b)
    q = pair_contact(s, s, b, a)
    assert np.allclose(r["force"], -q["force"], atol=1e-15)
    assert np.allclose(
        r["torque_i"] + r["torque_j"] + np.cross(a - b, r["force"]), 0, atol=1e-15
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"format": "bad"},
        {"particles": 0},
        {"timestep": float("nan")},
        {"friction": -1},
        {"stages": [{"kind": "unknown", "steps": 1}]},
        {"stages": [{"kind": "isotropic", "steps": 1}]},
    ],
)
def test_spec_rejects_invalid(kw):
    with pytest.raises(ValueError):
        Experiment(**kw)


def test_evidence_json_preserves_boolean_types(tmp_path):
    import json

    from sand_atlas_dem.util import write_json

    p = tmp_path / "report.json"
    write_json(
        p, {"passed": np.bool_(False), "count": np.int64(2), "values": np.array([1, 2])}
    )
    record = json.loads(p.read_text())
    assert record == {"passed": False, "count": 2, "values": [1, 2]}
    assert type(record["passed"]) is bool


def test_controlled_pair_workload_has_contacts_and_clearance():
    from sand_atlas_dem.benchmark import controlled_pairs
    from sand_atlas_dem.util import quat_matrix

    shape = sphere(resolution=24, samples=256)
    x, cell, ids, rotations, separations = controlled_pairs(ShapeLibrary([shape]), 8)
    assert np.all((x > 0) & (x < cell))
    assert np.array_equal(ids, np.zeros(8, dtype=np.uint32))
    for i in range(0, 8, 2):
        result = pair_contact(
            shape, shape, x[i], x[i + 1], qi=rotations[i], qj=rotations[i + 1]
        )
        assert result["contacts"] > 0 and np.linalg.norm(result["force"]) > 0
        assert np.allclose(
            (x[i + 1] - x[i]) @ quat_matrix(rotations[i]), [separations[0], 0, 0]
        )
        for j in range(8):
            if j // 2 != i // 2:
                delta = x[j] - x[i]
                delta -= np.round(delta / cell) * cell
                assert np.linalg.norm(delta) > 2 * shape.radius
