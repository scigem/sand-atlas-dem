"""Conservative ray/AABB pruning for the independent FP64 Trimesh oracle.

Trimesh 5.2 ray-triangle intersection, hit deduplication, ray directions and
parity rules are retained. Only impossible triangle candidates are removed
before Trimesh expands its per-ray triangle arrays. This is not a DEM backend.
"""

import numpy as np
from trimesh import grouping
from trimesh.ray.ray_triangle import RayMeshIntersector, ray_triangle_id


class RayBoxTree:
    """Adapter for Trimesh's ordered, one-query-per-ray candidate traversal."""

    def __init__(self, mesh, origins, directions):
        self.tree = mesh.triangles_tree
        self.bounds = self.tree.bounds
        triangles = mesh.triangles
        pad = 1e-5 * (1 + abs(mesh.vertices).max())
        self.lower = np.nextafter(triangles.min(axis=1) - pad, -np.inf)
        self.upper = np.nextafter(triangles.max(axis=1) + pad, np.inf)
        self.origins = np.asarray(origins, float)
        self.directions = np.asarray(directions, float)
        self.index = 0

    def intersection(self, bounds):
        index = self.index
        self.index += 1
        ids = np.fromiter(self.tree.intersection(bounds), dtype=np.int64)
        if not len(ids):
            return ids
        origin, direction = self.origins[index], self.directions[index]
        norm_squared = float(direction @ direction)
        if not np.isfinite(norm_squared) or norm_squared == 0:
            raise ValueError("Finite nonzero reference ray required")
        lo, hi = self.lower[ids], self.upper[ids]
        parallel = direction == 0
        valid = np.all(~parallel | ((lo <= origin) & (origin <= hi)), axis=1)
        near = np.full_like(lo, -np.inf)
        far = np.full_like(hi, np.inf)
        axes = ~parallel
        a = (lo[:, axes] - origin[axes]) / direction[axes]
        b = (hi[:, axes] - origin[axes]) / direction[axes]
        near[:, axes] = np.minimum(a, b)
        far[:, axes] = np.maximum(a, b)
        # Upstream accepts dot(location-origin, direction) > -1e-6.
        first = np.maximum(near.max(axis=1), -1e-6 / norm_squared)
        valid &= far.min(axis=1) >= first
        return ids[valid]


class BoundedRayIntersector(RayMeshIntersector):
    """Use upstream intersection and parity with a conservative candidate filter."""

    def intersects_id(
        self,
        ray_origins,
        ray_directions,
        return_locations=False,
        multiple_hits=True,
        **kwargs,
    ):
        tree = RayBoxTree(self.mesh, ray_origins, ray_directions)
        index_tri, index_ray, locations = ray_triangle_id(
            triangles=self.mesh.triangles,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            tree=tree,
            multiple_hits=multiple_hits,
            triangles_normal=self.mesh.face_normals,
        )
        if tree.index != len(ray_origins):
            raise ValueError("Upstream ray traversal no longer matches the adapter")
        if return_locations:
            if not len(index_tri):
                return index_tri, index_ray, locations
            unique = grouping.unique_rows(np.column_stack((locations, index_ray)))[0]
            return index_tri[unique], index_ray[unique], locations[unique]
        return index_tri, index_ray
