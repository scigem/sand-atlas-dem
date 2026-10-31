"""Topology and packed bounds are verified independently of GPU traversal."""

from dataclasses import replace

import numpy as np
import pytest
import trimesh

from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.mesh import pack_meshes


@pytest.fixture
def box():
    return from_mesh(trimesh.creation.box(), unit="mm", resolution=8, samples=32)


def test_packed_tree_covers_original_triangles_and_bounds(box):
    data = pack_meshes(ShapeLibrary([box, replace(box, id="box-2")]), 0.001)
    triangles = data["meshTriangles"][:, :3].reshape(-1, 3, 3)
    links = data["meshLinks"]
    for root, end, _, _ in data["meshRoots"]:
        visits = []
        node = int(root)
        while node < end:
            escape, start, count, _ = links[node]
            assert node < escape <= end
            if count:
                leaf = triangles[start : start + count]
                assert np.all(leaf >= data["meshMin"][node, :3])
                assert np.all(leaf <= data["meshMax"][node, :3])
                visits.extend(range(int(start), int(start + count)))
            node += 1
        assert len(visits) == len(box.faces)
    assert len(triangles) == 2 * len(box.faces)
    for root in range(2):
        for start, count, _, _ in data["meshHullInfo"][6 * root : 6 * root + 6]:
            plane = data["meshPlanes"][start : start + count]
            assert np.all((box.vertices / 0.001) @ plane[:, :3].T + plane[:, 3] < 0)
    assert np.allclose(np.linalg.norm(data["meshNormals"][:, :3], axis=1), 1)
    assert np.allclose(
        np.sort(triangles.reshape(-1)),
        np.sort(np.tile(box.vertices[box.faces].ravel() / 0.001, 2)),
    )


def test_open_or_inconsistently_oriented_mesh_is_rejected(box):
    with pytest.raises(ValueError, match="two faces"):
        pack_meshes(ShapeLibrary([replace(box, faces=box.faces[:-1])]), 0.001)
    faces = box.faces.copy()
    faces[0] = faces[0, ::-1]
    with pytest.raises(ValueError, match="winding"):
        pack_meshes(ShapeLibrary([replace(box, faces=faces)]), 0.001)


@pytest.mark.parametrize("count", [1, 32, 64, 65, 129, 1024])
def test_spatial_clusters_preserve_every_original_node_key_and_enclose_points(
    box, count
):
    rng = np.random.default_rng(321)
    # Arbitrary points exercise uneven partitions, not just surface regularity.
    points = rng.normal(size=(count, 3)) * 0.001
    shape = replace(box, samples=points, areas=np.ones(count))
    data = pack_meshes(ShapeLibrary([shape, replace(shape, id="second")]), 0.001)
    for first, clusters, _, _ in data["meshClusterInfo"]:
        keys = []
        for cluster in range(first, first + clusters):
            start, size, _, _ = data["meshClusterRanges"][cluster]
            assert 0 < size <= 64
            node_ids = data["meshSampleIndices"][start : start + size]
            keys.extend(node_ids.tolist())
            rounded = (points[node_ids] / 0.001).astype(np.float32)
            assert np.all(rounded >= data["meshClusterMin"][cluster, :3])
            assert np.all(rounded <= data["meshClusterMax"][cluster, :3])
        assert sorted(keys) == list(range(count))


def test_disconnected_vertex_fans_are_rejected(box):
    # Two otherwise closed tetrahedra share exactly one vertex index.
    a = trimesh.creation.icosphere(subdivisions=0)
    b = a.copy()
    b.apply_translation(a.vertices[0] - b.vertices[0] + [0, 0, 4])
    b.vertices[0] = a.vertices[0]
    vertices = np.vstack([a.vertices, b.vertices[1:]])
    remap = np.r_[0, np.arange(len(a.vertices), len(vertices))]
    faces = np.vstack([a.faces, remap[b.faces]])
    with pytest.raises(ValueError, match="vertex fans"):
        pack_meshes(ShapeLibrary([replace(box, vertices=vertices, faces=faces)]), 1)


def test_display_only_mesh_cannot_replace_analytical_geometry():
    from sand_atlas_dem.geometry import sphere

    with pytest.raises(ValueError, match="display-only"):
        pack_meshes(ShapeLibrary([sphere(resolution=8, samples=32)]), 0.001)


def test_face_quadrature_integrates_known_surface_polynomial(box):
    from sand_atlas_dem.geometry import with_face_quadrature

    original_samples = box.samples.copy()
    sampled = with_face_quadrature(box)
    assert len(sampled.samples) == 3 * len(box.faces)
    assert sampled.vertices is box.vertices and sampled.sdf is box.sdf
    assert np.isclose(sampled.areas.sum(), 6e-6, rtol=1e-12)
    # On a cube of side a: two x faces give a^4/2, four others a^4/3.
    assert np.isclose(
        np.sum(sampled.areas * sampled.samples[:, 0] ** 2),
        5 / 6 * 1e-12,
        rtol=1e-12,
        atol=1e-25,
    )
    assert np.array_equal(box.samples, original_samples)
    assert sampled.provenance["sampling"] == "source_triangle_gauss3"
