"""Deterministic bidirectional parity, with direct winding for disagreements."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np


def _load(name, filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rays = _load("inside_reference_rays", "reference_rays.py")
_winding = _load("inside_reference_winding", "reference_winding.py")


class DeterministicRayIntersector(_rays.BoundedRayIntersector):
    """Retain default primary rays; never make an entropy-seeded fallback draw.

    Agreeing forward/backward parities are accepted. Every disagreement uses
    an exhaustive oriented solid-angle sum over the original target triangles.
    A sum outside the declared neighborhoods of zero and one rejects the query.
    """

    def __init__(self, mesh, winding_tolerance=1e-8):
        super().__init__(mesh)
        self.winding_tolerance = winding_tolerance
        self.winding_checks = 0
        self.maximum_integer_residual = 0.0

    def contains_points(self, points):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Finite (n,3) query points required")
        result = np.zeros(len(points), bool)
        keep = np.all(
            (points >= self.mesh.bounds[0]) & (points <= self.mesh.bounds[1]), axis=1
        )
        ids = np.flatnonzero(keep)
        if not len(ids):
            return result
        p = points[ids]
        direction = np.tile([0.4395064455, 0.617598629942, 0.652231566745], (len(p), 1))
        _, ray_index, _ = self.intersects_location(
            np.vstack([p, p]), np.vstack([direction, -direction]), multiple_hits=True
        )
        hits = np.bincount(ray_index, minlength=2 * len(p)).reshape(2, -1)
        parity = (hits % 2) == 1
        agree = parity[0] == parity[1]
        result[ids[agree]] = parity[0, agree]
        for local in np.flatnonzero(~agree):
            w = _winding.winding_number(self.mesh.triangles, p[local])
            residual = min(abs(w), abs(w - 1))
            self.winding_checks += 1
            self.maximum_integer_residual = max(self.maximum_integer_residual, residual)
            if not np.isfinite(w) or residual > self.winding_tolerance:
                key = hashlib.sha256(p[local].tobytes()).hexdigest()
                raise ValueError(
                    f"Unresolved winding {w} at point {key}; no surface/solid classification inferred"
                )
            result[ids[local]] = abs(w - 1) <= self.winding_tolerance
        return result
