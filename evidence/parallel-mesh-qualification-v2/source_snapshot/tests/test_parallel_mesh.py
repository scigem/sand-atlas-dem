"""Frozen parallel/serial mesh coverage and force/history equivalence gates."""

import os
from dataclasses import replace

import numpy as np
import pytest
import trimesh

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh, with_face_quadrature
from sand_atlas_dem.gpu import Engine, f4
from sand_atlas_dem.mesh import pack_meshes
from sand_atlas_dem.spec import Experiment


@pytest.mark.parametrize("count", [1, 64, 65, 1024, 8193])
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
    assert len(data["meshFrontier"]) == 32


class SerialEngine(Engine):
    """Original collector; identical current force law and integrator."""

    def _encode_contacts(self, encoder, bindings):
        self.dispatch("resetMeshContacts", self.pair_capacity, encoder, bindings)
        self.dispatch("collectMeshContacts", 64 * self.pair_capacity, encoder, bindings)
        if self.wall_count:
            self.dispatch(
                "collectWallContacts", 64 * self.n * self.wall_count, encoder, bindings
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
