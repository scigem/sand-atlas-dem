"""Frozen parallel/serial mesh coverage and force/history equivalence gates."""

import os
from dataclasses import replace

import numpy as np
import pytest
import trimesh

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh, with_face_quadrature
from sand_atlas_dem.gpu import Engine, f4
from sand_atlas_dem.mesh import mesh_frontier_count, pack_meshes
from sand_atlas_dem.spec import Experiment


@pytest.mark.parametrize("count", [1, 64, 65, 1024, 8193, 131072])
def test_frontiers_partition_original_keys(count):
    shape = from_mesh(trimesh.creation.box(), unit="mm", resolution=8, samples=16)
    rng = np.random.default_rng(592)
    shape = replace(
        shape, samples=rng.normal(size=(count, 3)) * 0.001, areas=np.ones(count)
    )
    data = pack_meshes(ShapeLibrary([shape]), 0.001)
    keys = []
    for first, end, _, _ in data["meshFrontier"]:
        if end == 0:
            continue
        assert end == data["meshClusterRanges"][first, 2]
        for node in range(first, end):
            start, size, _, _ = data["meshClusterRanges"][node]
            keys.extend(data["meshSampleIndices"][start : start + size])
    assert sorted(map(int, keys)) == list(range(count))
    assert len(data["meshFrontier"]) == (128 if count >= 131072 else 32)


class SerialEngine(Engine):
    """Original collector; identical current force law and integrator."""

    def _encode_contacts(self, encoder, bindings):
        self.dispatch("resetMeshContacts", self.pair_capacity, encoder, bindings)
        self.dispatch("collectMeshContacts", 64 * self.pair_capacity, encoder, bindings)
        if self.wall_count:
            self.dispatch(
                "collectWallContacts", 64 * self.n * self.wall_count, encoder, bindings
            )


def test_fine_frontier_uses_median_source_resolution():
    small = from_mesh(trimesh.creation.box(), unit="mm", resolution=8, samples=16)
    fine = replace(
        small,
        id="fine",
        samples=np.tile(small.samples, (8192, 1)),
        areas=np.tile(small.areas, 8192) / 8192,
    )
    assert mesh_frontier_count(ShapeLibrary([fine])) == 128
    assert mesh_frontier_count(ShapeLibrary([small, fine])) == 32
    assert (
        mesh_frontier_count(ShapeLibrary([small, fine, replace(fine, id="fine2")]))
        == 128
    )
    assert (
        mesh_frontier_count(
            ShapeLibrary(
                [replace(fine, samples=fine.samples[:-1], areas=fine.areas[:-1])]
            )
        )
        == 32
    )


@pytest.mark.metal
@pytest.mark.skipif(
    os.environ.get("SAND_DEM_METAL") != "1", reason="Serial Metal required"
)
def test_production_fine_collector_matches_serial_static_contact():
    shape = with_face_quadrature(
        from_mesh(
            trimesh.creation.box(extents=[100] * 3), unit="um", resolution=8, samples=32
        ),
        rule="gauss3",
        subdivisions=6,
    )
    metal = Metal()
    states = []
    for cls in [SerialEngine, Engine]:
        e = cls(
            ShapeLibrary([shape]),
            Experiment(
                periodic=False,
                contact_geometry="mesh",
                particles=2,
                timestep=2e-9,
                contacts_per_particle=131072,
                pairs_per_particle=4,
                bin_capacity=16,
            ),
            [[0.003] * 3, [0.003 + 0.999e-4, 0.003, 0.003]],
            [0.01] * 3,
            [0, 0],
            [1, 1],
            metal=metal,
        )
        assert e.mesh_frontier_count == 128
        e.initialize()
        states.append(e.state())
    a, b = states
    assert a["contacts"] > 40000
    ai = np.lexsort(a["meta"][:, :3].T[::-1])
    bi = np.lexsort(b["meta"][:, :3].T[::-1])
    np.testing.assert_array_equal(a["meta"][ai], b["meta"][bi])
    np.testing.assert_allclose(
        a["normalArea"][ai], b["normalArea"][bi], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        a["force"][:, :3], b["force"][:, :3], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        a["torque"][:, :3], b["torque"][:, :3], rtol=1e-6, atol=1e-7
    )


@pytest.mark.metal
@pytest.mark.skipif(
    os.environ.get("SAND_DEM_METAL") != "1", reason="Serial Metal required"
)
@pytest.mark.parametrize("concave", [False, True])
def test_parallel_serial_force_history_and_motion(concave):
    mesh = (
        trimesh.creation.annulus(r_min=20, r_max=50, height=40, sections=24)
        if concave
        else trimesh.creation.box(extents=[100, 70, 40])
    )
    sh = with_face_quadrature(
        from_mesh(mesh, unit="um", resolution=8, samples=32),
        rule="gauss3",
        subdivisions=2,
    )
    metal = Metal()
    states = []
    for cls in [SerialEngine, Engine]:
        spec = Experiment(
            periodic=False,
            contact_geometry="mesh",
            particles=2,
            timestep=2e-9,
            max_overlap_fraction=0.3,
            contacts_per_particle=4096,
            pairs_per_particle=4,
            bin_capacity=16,
        )
        e = cls(
            ShapeLibrary([sh]),
            spec,
            [[0.003, 0.003, 0.003], [0.003 + 8.5e-5, 0.003 + 2e-6, 0.003]],
            [0.01] * 3,
            [0, 0],
            [1, 1],
            metal=metal,
        )
        angle = 0.1
        e.buffers["rotation"].copy_from_numpy(
            np.array(
                [[0, 0, 0, 1], [0, 0, np.sin(angle / 2), np.cos(angle / 2)]], np.float32
            )
        )
        e.buffers["velocity"].copy_from_numpy(
            f4([[0, 0.01, 0], [0, -0.01, 0]]) * e.time_unit / e.length_unit
        )
        e.initialize()
        assert e.state()["contacts"] > 0
        initial = e.state()
        e.advance(16)
        final = e.state()
        e.restore(initial, 0)
        e.advance(16)
        repeated = e.state()
        states.append((initial, final, repeated))
    for first, second in zip(states[0], states[1]):
        for name in ["position", "velocity", "rotation"]:
            assert np.max(abs(first[name] - second[name])) <= 1e-5
        for name in ["force", "torque"]:
            scale = max(
                float(np.max(np.linalg.norm(first[name][:, :3], axis=1))), 1e-12
            )
            assert (
                np.max(np.linalg.norm(first[name][:, :3] - second[name][:, :3], axis=1))
                / scale
                <= 1e-4
            )
        a = np.lexsort(first["meta"][:, :3].T[::-1])
        b = np.lexsort(second["meta"][:, :3].T[::-1])
        np.testing.assert_array_equal(first["meta"][a], second["meta"][b])
        np.testing.assert_allclose(
            first["normalArea"][a], second["normalArea"][b], rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            first["slip"][a], second["slip"][b], rtol=1e-4, atol=1e-8
        )


@pytest.mark.metal
@pytest.mark.skipif(
    os.environ.get("SAND_DEM_METAL") != "1", reason="Serial Metal required"
)
@pytest.mark.parametrize(
    "seed,scales", [(531, [1.0, 0.7]), (532, [4.0, 0.2]), (533, [0.2, 4.0])]
)
def test_hoisted_bounds_enclose_independent_fp64_corners(seed, scales):
    shape = from_mesh(
        trimesh.creation.box(extents=[100, 70, 40]),
        unit="um",
        resolution=8,
        samples=1024,
    )
    metal = Metal()
    e = Engine(
        ShapeLibrary([shape]),
        Experiment(
            periodic=False,
            particles=2,
            contact_geometry="mesh",
            timestep=1e-10,
            pairs_per_particle=4,
        ),
        [[0.003] * 3, [0.003] * 3],
        [0.01] * 3,
        [0, 0],
        scales,
        metal=metal,
    )
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(2, 4))
    q /= np.linalg.norm(q, axis=1)[:, None]
    q = q.astype(np.float32)
    e.buffers["rotation"].copy_from_numpy(q)
    control = np.zeros(16, np.uint32)
    control[1] = 1
    e.buffers["control"].copy_from_numpy(control)
    e.m.write_prefix(e.buffers["pairs"], np.array([[0, 1]], np.uint32))
    e.m.dispatch("prepareMeshPairs", e.pair_capacity, r=e.bindings(dt=0))
    e.m.device.wait()
    assert e.m.read(e.buffers["meshPairMask"], np.uint32)[0] == 3
    count = e.buffers["meshClusterRanges"].size // 16
    low = e.m.read(e.buffers["meshClusterMin"], shape=(count, 4))[:, :3].astype(float)
    high = e.m.read(e.buffers["meshClusterMax"], shape=(count, 4))[:, :3].astype(float)
    corners = np.stack(
        [np.where(np.array([c & 1, c & 2, c & 4], bool), high, low) for c in range(8)],
        axis=1,
    )
    result = e.m.buffer(count=8 * count)

    def rotate(q, v):
        return v + 2 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)

    for direction in range(2):
        e.m.dispatch(
            "meshPairBoundsProbe",
            count,
            r=e.bindings(dt=0),
            count=count,
            direction=direction,
            first=0,
            result=result,
        )
        e.m.device.wait()
        bounds = e.m.read(result, shape=(count, 2, 4))[:, :, :3].astype(float)
        src, dst = (0, 1) if direction == 0 else (1, 0)
        conjugate = q[dst].astype(float) * [-1, -1, -1, 1]
        exact = rotate(
            conjugate, rotate(q[src].astype(float), corners * float(e.body[src, 0]))
        ) / float(e.body[dst, 0])
        assert np.all(exact >= bounds[:, 0, None, :])
        assert np.all(exact <= bounds[:, 1, None, :])
