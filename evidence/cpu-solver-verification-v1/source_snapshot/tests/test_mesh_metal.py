"""Frozen analytical distance bounds exercise face, edge, vertex and concave queries."""

import os

import numpy as np
import pytest
import trimesh

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.gpu import Engine, f4
from sand_atlas_dem.spec import Experiment

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(
        os.environ.get("SAND_DEM_METAL") != "1",
        reason="Explicit serial Metal execution required",
    ),
]


def test_mesh_distance_matches_analytical_box_and_packed_shape_offsets():
    from dataclasses import replace

    shape = from_mesh(
        trimesh.creation.box(extents=[1, 2, 3]), unit="mm", resolution=8, samples=128
    )
    other = replace(shape, id="another-box")
    library = ShapeLibrary([shape, other])
    m = Metal()
    spec = Experiment(particles=2, timestep=1e-9, contact_geometry="mesh")
    e = Engine(
        library,
        spec,
        [[0.005] * 3, [0.015] * 3],
        [0.03] * 3,
        [0, 1],
        [1.0, 1.0],
        metal=m,
    )
    rng = np.random.default_rng(79)
    points = rng.uniform(-0.002, 0.002, (513, 3))
    points = np.vstack(
        [
            points,
            [
                [0.0006, 0, 0],
                [0.0006, 0.0011, 0],
                [0.0006, 0.0011, 0.0016],
                [0, 0, 0],
                [0.00049, 0, 0],
            ],
        ]
    )
    q = np.abs(points) - [0.0005, 0.001, 0.0015]
    expected = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    a = m.buffer(f4(points / e.length_unit))
    b = m.buffer(count=len(points), stride=16)
    for index in (0, 1):
        m.dispatch(
            "meshProbe",
            len(points),
            r=e.bindings(),
            count=len(points),
            shape=index,
            points=a,
            result=b,
        )
        m.device.wait()
        result = m.read(b).reshape(-1, 4)
        assert np.max(abs(result[:, 3] * e.length_unit - expected)) < 2e-9
        assert np.allclose(np.linalg.norm(result[:, :3], axis=1), 1, atol=2e-5)
        # At an outside edge and corner, the distance gradient follows the offset.
        assert np.allclose(result[-4, :3], [2**-0.5, 2**-0.5, 0], atol=2e-5)
        assert np.allclose(result[-3, :3], np.full(3, 3**-0.5), atol=2e-5)


def test_mesh_concave_distance_against_independent_triangle_queries():
    from pathlib import Path

    library = ShapeLibrary.load(Path("assets/generated/concave-prism.h5"))
    shape = library.shapes[0]
    m = Metal()
    spec = Experiment(particles=2, timestep=1e-9, contact_geometry="mesh")
    e = Engine(
        library,
        spec,
        [[0.002] * 3, [0.005] * 3],
        [0.008] * 3,
        [0, 0],
        [1.0, 1.0],
        metal=m,
    )
    rng = np.random.default_rng(93)
    points = rng.uniform(
        shape.vertices.min(0) * 1.2, shape.vertices.max(0) * 1.2, (1024, 3)
    )
    mesh = trimesh.Trimesh(shape.vertices / e.length_unit, shape.faces, process=False)
    expected = (
        -trimesh.proximity.signed_distance(mesh, points / e.length_unit) * e.length_unit
    )
    a = m.buffer(f4(points / e.length_unit))
    b = m.buffer(count=len(points), stride=16)
    m.dispatch(
        "meshProbe",
        len(points),
        r=e.bindings(),
        count=len(points),
        shape=0,
        points=a,
        result=b,
    )
    m.device.wait()
    result = m.read(b).reshape(-1, 4)
    assert np.max(abs(result[:, 3] * e.length_unit - expected)) < 2e-9
    assert np.array_equal(result[:, 3] < 0, expected < 0)
    enclosed = m.buffer(count=len(points))
    m.dispatch(
        "meshCullProbe",
        len(points),
        r=e.bindings(),
        count=len(points),
        shape=0,
        points=a,
        result=enclosed,
    )
    m.device.wait()
    assert np.all(m.read(enclosed, np.uint32)[expected < 0] == 1)


def test_mesh_compaction_growth_and_sample_key_friction_restart():
    shape = from_mesh(trimesh.creation.box(), unit="mm", resolution=8, samples=1024)
    m = Metal()
    spec = Experiment(
        particles=2,
        timestep=1e-9,
        contact_geometry="mesh",
        contacts_per_particle=16,
        pairs_per_particle=4,
    )
    origin = np.full(3, 0.005)
    positions = np.array([origin, origin + [0.00102, 0, 0]])
    e = Engine(
        ShapeLibrary([shape]), spec, positions, [0.02] * 3, [0, 0], [1.0, 1.0], metal=m
    )
    e.initialize()
    assert e.diagnostics()["active_surface_contacts"] == 0
    # Cross into a new flat-face contact through an actual drift. The committed
    # positions/contact state remain consistent for every rollback and resize.
    initial_capacity = e.contact_capacity
    approach = np.zeros((2, 4), np.float32)
    approach[1, 0] = -0.00003 / spec.timestep * e.time_unit / e.length_unit
    e.buffers["velocity"].copy_from_numpy(approach)
    e.advance(1)
    diagnostics = e.diagnostics()
    assert diagnostics["active_grain_pairs"] == 1
    count = diagnostics["active_surface_contacts"]
    assert (
        count > initial_capacity and e.contact_capacity >= count and e.capacity_retries
    )
    # Exact box distance includes edge-near points whose nearest surface is a
    # side face. Treating the whole overlap as a single plane is incorrect.
    from dataclasses import replace

    from sand_atlas_dem.reference import pair_contact

    analytic = replace(shape)

    def box_query(points):
        q = np.abs(points) - 0.0005
        outside = np.maximum(q, 0)
        length = np.linalg.norm(outside, axis=1)
        phi = length + np.minimum(q.max(axis=1), 0)
        gradient = np.zeros_like(points)
        far = length > 0
        gradient[far] = np.sign(points[far]) * outside[far] / length[far, None]
        rows = np.flatnonzero(~far)
        axes = q[~far].argmax(axis=1)
        gradient[rows, axes] = np.sign(points[rows, axes])
        return phi, gradient

    analytic.query = box_query
    expected = pair_contact(
        analytic,
        analytic,
        origin,
        origin + [0.00099, 0, 0],
        kn=spec.normal_stiffness,
        kt=spec.tangential_stiffness,
        mu=spec.friction,
        vj=[-0.00003 / spec.timestep, 0, 0],
        dt=spec.timestep,
    )["force"]
    force = m.read(e.buffers["force"]).reshape(-1, 4)[0, :3] * e.force_unit
    assert np.linalg.norm(force - expected) / np.linalg.norm(expected) < 1e-4
    v = np.zeros((2, 4), np.float32)
    v[0, 1] = 0.001
    e.buffers["velocity"].copy_from_numpy(v)
    e.advance(3)
    saved = e.state()
    e.advance(2)
    first = e.state()
    e.restore(saved, e.step - 2)
    e.advance(2)
    second = e.state()

    def slips(state):
        return {
            tuple(key[:3]): value[:3]
            for key, value in zip(state["meta"], state["slip"])
        }

    a, b = slips(first), slips(second)
    assert a.keys() == b.keys()
    assert any(np.linalg.norm(s) > 0 for s in a.values())
    assert all(np.allclose(a[key], b[key], rtol=1e-4, atol=1e-9) for key in a)


def test_cluster_culling_retains_all_analytical_inside_nodes_under_rotation_and_scale():
    from scipy.spatial.transform import Rotation

    shape = from_mesh(trimesh.creation.box(), unit="mm", resolution=8, samples=1024)
    m = Metal()
    e = Engine(
        ShapeLibrary([shape]),
        Experiment(particles=2, timestep=1e-9, contact_geometry="mesh"),
        [[0.005] * 3, [0.008] * 3],
        [0.02] * 3,
        [0, 0],
        [0.8, 1.3],
        metal=m,
    )
    info = m.read(e.buffers["meshClusterInfo"], np.uint32).reshape(-1, 4)[0]
    first, count = map(int, info[:2])
    ranges = m.read(e.buffers["meshClusterRanges"], np.uint32).reshape(-1, 4)
    indices = m.read(e.buffers["meshSampleIndices"], np.uint32)
    result = m.buffer(count=count)
    rng = np.random.default_rng(518)
    rejected = penetrating = 0
    for _ in range(24):
        rotations = Rotation.random(2, random_state=rng)
        source = np.full(3, 0.005)
        target = source + rotations[1].apply([0.0009, 0, 0])
        e.buffers["position"].copy_from_numpy(
            f4(np.array([source, target]) / e.length_unit)
        )
        e.buffers["rotation"].copy_from_numpy(rotations.as_quat().astype(np.float32))
        local = (
            rotations[1]
            .inv()
            .apply(rotations[0].apply(shape.samples * 0.8) + source - target)
            / 1.3
        )
        inside = np.all(np.abs(local) < 0.0005, axis=1)
        penetrating += int(inside.sum())
        m.dispatch(
            "meshClusterProbe",
            count,
            r=e.bindings(),
            count=count,
            src=0,
            dst=1,
            result=result,
        )
        m.device.wait()
        keep = m.read(result, np.uint32)
        rejected += int((keep == 0).sum())
        for cluster in range(count):
            start, size = map(int, ranges[first + cluster, :2])
            if size:
                keys = indices[start : start + size]
            else:
                # A rejected ancestor would hide every descendant from the
                # stackless collector, even if its individual leaf is safe.
                escape = int(ranges[first + cluster, 2])
                keys = np.concatenate(
                    [
                        indices[
                            int(ranges[node, 0]) : int(
                                ranges[node, 0] + ranges[node, 1]
                            )
                        ]
                        for node in range(first + cluster + 1, escape)
                        if ranges[node, 1] > 0
                    ]
                )
            if inside[keys].any():
                assert keep[cluster] == 1
    assert penetrating > 0 and rejected > 0
