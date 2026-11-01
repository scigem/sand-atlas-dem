"""Shared static triangle hierarchies for exact-feature GPU distance queries.

Sign: Baerentzen & Aanaes (2005), DOI 10.1109/TVCG.2005.49, eqs. (3)-(5).
Outward closed orientable two-manifold inputs; no repair or convex decomposition.
"""

import numpy as np


def mesh_frontier_count(library):
    """Fine libraries expose more complete subtrees; mixed/coarse libraries retain32."""
    return 128 if min(len(s.samples) for s in library.shapes) >= 131072 else 32


def pack_meshes(library, length_unit):
    bounds_min, bounds_max, links, roots, triangles, normals = [], [], [], [], [], []
    hull_info, planes = [], []
    frontier = []
    frontier_count = mesh_frontier_count(library)
    cluster_info, cluster_ranges, cluster_min, cluster_max, sample_indices = (
        [],
        [],
        [],
        [],
        [],
    )
    for shape in library.shapes:
        if shape.provenance.get("display_mesh_only"):
            raise ValueError(
                "Mesh query requires physical source triangles, not a display-only fixture mesh"
            )
        vertices = np.asarray(shape.vertices, float) / length_unit
        faces = np.asarray(shape.faces, np.int64)
        # Spatial blocks preserve original node keys through an index map. Bounds
        # enclose the actual rounded GPU sample coordinates, not source vertices.
        points = (shape.samples / length_unit).astype(np.float32)
        first_cluster = len(cluster_ranges)

        def partition(ids, points=points):
            node = len(cluster_ranges)
            cluster_ranges.append([0, 0, 0, 0])
            cluster_min.append(
                [*np.nextafter(points[ids].min(0), np.float32(-np.inf)), 0]
            )
            cluster_max.append(
                [*np.nextafter(points[ids].max(0), np.float32(np.inf)), 0]
            )
            if len(ids) > 64:
                axis = np.ptp(points[ids], axis=0).argmax()
                order = ids[np.argsort(points[ids, axis], kind="stable")]
                middle = len(order) // 2
                partition(order[:middle])
                partition(order[middle:])
            else:
                cluster_ranges[node][:2] = [len(sample_indices), len(ids)]
                sample_indices.extend(ids.tolist())
            # Preorder layout: skip the entire subtree through its escape link.
            # A zero sample count marks an internal node, never a surface key.
            cluster_ranges[node][2] = len(cluster_ranges)

        partition(np.arange(len(points)))
        cluster_info.append([first_cluster, len(cluster_ranges) - first_cluster, 0, 0])
        # Disjoint complete subtrees cover every original sample exactly once.
        # Split the largest remaining subtree first; small shapes use fewer slots.
        pending = [first_cluster]
        while len(pending) < frontier_count:
            splittable = [node for node in pending if cluster_ranges[node][1] == 0]
            if not splittable:
                break
            node = max(splittable, key=lambda n: cluster_ranges[n][2] - n)
            left = node + 1
            right = cluster_ranges[left][2]
            at = pending.index(node)
            pending[at : at + 1] = [left, right]
        frontier.extend([[node, cluster_ranges[node][2], 0, 0] for node in pending])
        frontier.extend([[0, 0, 0, 0]] * (frontier_count - len(pending)))
        # Conservative enclosure only: original concave triangles still determine
        # every penetrating distance and normal. Round offsets outward so all
        # GPU-rounded source vertices remain inside even under FP32 dot error.
        from scipy.spatial import ConvexHull

        rounded_vertices = vertices.astype(np.float32).astype(float)
        hull = ConvexHull(rounded_vertices)
        margin = 16 * np.finfo(np.float32).eps * (1 + abs(rounded_vertices).max())
        hull_planes = hull.equations.astype(np.float32)
        hull_planes[:, 3] = np.nextafter(
            (hull.equations[:, 3] - margin).astype(np.float32), np.float32(-np.inf)
        )
        hull_planes = np.unique(hull_planes, axis=0)
        directions = hull_planes[:, :3]
        dominant = np.abs(directions).argmax(axis=1)
        groups = 2 * dominant + (directions[np.arange(len(directions)), dominant] >= 0)
        for group in range(6):
            subset = hull_planes[groups == group]
            hull_info.append([len(planes), len(subset), 0, 0])
            planes.extend(subset)
        tri = vertices[faces]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(cross, axis=1)
        if np.any(norm <= 0):
            raise ValueError("Degenerate source triangle")
        face_normal = cross / norm[:, None]
        angles = []
        for corner in range(3):
            a = tri[:, (corner + 1) % 3] - tri[:, corner]
            b = tri[:, (corner + 2) % 3] - tri[:, corner]
            angles.append(
                np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), (a * b).sum(1))
            )
        angle = np.stack(angles, axis=1)
        vertex_normal = np.zeros_like(vertices)
        np.add.at(
            vertex_normal,
            faces.ravel(),
            (face_normal[:, None, :] * angle[:, :, None]).reshape(-1, 3),
        )
        edges = faces[:, [[0, 1], [1, 2], [2, 0]]]
        unique, inverse, count = np.unique(
            np.sort(edges.reshape(-1, 2), axis=1),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        if np.any(count != 2):
            raise ValueError("Mesh query requires exactly two faces per edge")
        # Opposite directed edge orientations are required, independently of volume.
        direction = np.where(
            edges.reshape(-1, 2)[:, 0] < edges.reshape(-1, 2)[:, 1], 1, -1
        )
        oriented = np.zeros(len(unique), int)
        np.add.at(oriented, inverse, direction)
        if np.any(oriented):
            raise ValueError("Inconsistent source winding")
        # Every vertex link must be one cycle, not disconnected fans touching at
        # a vertex. Two-face edges alone do not establish a two-manifold.
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        occurrence = np.arange(3 * len(faces))
        corner_a = 3 * (occurrence // 3) + occurrence % 3
        corner_b = 3 * (occurrence // 3) + (occurrence % 3 + 1) % 3
        corners = np.stack([corner_a, corner_b], axis=1)
        corners[direction < 0] = corners[direction < 0, ::-1]
        paired = corners[np.argsort(inverse, kind="stable")].reshape(-1, 2, 2)
        row = paired[:, 0].ravel()
        col = paired[:, 1].ravel()
        graph = coo_matrix((np.ones(len(row)), (row, col)), shape=(3 * len(faces),) * 2)
        components, _ = connected_components(graph, directed=False)
        if components != len(np.unique(faces)):
            raise ValueError("Disconnected vertex fans in source mesh")
        edge_normal = np.zeros((len(unique), 3))
        np.add.at(edge_normal, inverse, np.repeat(face_normal, 3, axis=0))
        feature = np.concatenate(
            [
                face_normal[:, None, :],
                edge_normal[inverse.reshape(-1, 3)],
                vertex_normal[faces],
            ],
            axis=1,
        )
        lengths = np.linalg.norm(feature, axis=2)
        if np.any(lengths < 1e-12):
            raise ValueError("Undefined mesh feature pseudonormal")
        feature /= lengths[:, :, None]
        # Bounds enclose the same FP32 triangles used by the GPU.
        tri = tri.astype(np.float32)
        rounded_cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if np.any(np.linalg.norm(rounded_cross, axis=1) == 0):
            raise ValueError("Triangle degenerates in normalized FP32")
        root = len(links)
        first_triangle = len(triangles)
        ordered = []
        depths = []

        def build(
            ids,
            depth=0,
            tri=tri,
            first_triangle=first_triangle,
            ordered=ordered,
            depths=depths,
        ):
            node = len(links)
            depths.append(depth)
            lo = np.nextafter(tri[ids].min(axis=(0, 1)), np.float32(-np.inf))
            hi = np.nextafter(tri[ids].max(axis=(0, 1)), np.float32(np.inf))
            bounds_min.append([*lo, 0])
            bounds_max.append([*hi, 0])
            links.append([0, 0, 0, 0])
            if len(ids) <= 4:
                links[node][1:3] = [first_triangle + len(ordered), len(ids)]
                ordered.extend(ids.tolist())
            else:
                centers = tri[ids].mean(axis=1)
                axis = np.ptp(centers, axis=0).argmax()
                ids = ids[np.argsort(centers[:, axis], kind="stable")]
                middle = len(ids) // 2
                build(ids[:middle], depth + 1)
                build(ids[middle:], depth + 1)
            links[node][0] = len(links)  # Stackless escape index, including leaf.

        build(np.arange(len(tri)))
        if max(depths) > 30:
            raise ValueError("Mesh hierarchy exceeds GPU traversal stack depth")
        roots.append([root, len(links), max(depths), 0])
        triangles.extend(tri[ordered])
        normals.extend(feature[ordered])

    def f4(a):
        a = np.asarray(a, np.float32).reshape(-1, 3)
        return np.pad(a, ((0, 0), (0, 1)))

    return {
        "meshFrontier": np.asarray(frontier, np.uint32),
        "meshClusterInfo": np.asarray(cluster_info, np.uint32),
        "meshClusterRanges": np.asarray(cluster_ranges, np.uint32),
        "meshClusterMin": np.asarray(cluster_min, np.float32),
        "meshClusterMax": np.asarray(cluster_max, np.float32),
        "meshSampleIndices": np.asarray(sample_indices, np.uint32),
        "meshHullInfo": np.asarray(hull_info, np.uint32),
        "meshPlanes": np.asarray(planes, np.float32),
        "meshRoots": np.asarray(roots, np.uint32),
        "meshMin": np.asarray(bounds_min, np.float32),
        "meshMax": np.asarray(bounds_max, np.float32),
        "meshLinks": np.asarray(links, np.uint32),
        "meshTriangles": f4(triangles),
        "meshNormals": f4(normals),
    }
