"""Direct oriented solid-angle sum for disputed CPU reference points.

Formula checked against libigl's solid_angle and winding_number definitions:
https://github.com/libigl/libigl/blob/main/include/igl/solid_angle.cpp
Independent NumPy evaluation; no libigl code or binary is used. This diagnostic
does not repair geometry or silently decide how to classify surface points.
"""

import math

import numpy as np


def winding_number(triangles, point, batch=16384):
    triangles = np.asarray(triangles, float)
    point = np.asarray(point, float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or point.shape != (3,):
        raise ValueError("Expected triangles (n,3,3) and one point (3,)")
    if not np.isfinite(triangles).all() or not np.isfinite(point).all():
        raise ValueError("Finite source geometry and query required")
    sums = []
    for start in range(0, len(triangles), batch):
        vectors = triangles[start : start + batch] - point
        a, b, c = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        lengths = np.linalg.norm(vectors, axis=2)
        if np.any(lengths == 0):
            raise ValueError("Winding number is undefined at a source vertex")
        determinant = np.sum(a * np.cross(b, c), axis=1)
        denominator = (
            lengths.prod(axis=1)
            + np.sum(a * b, axis=1) * lengths[:, 2]
            + np.sum(b * c, axis=1) * lengths[:, 0]
            + np.sum(c * a, axis=1) * lengths[:, 1]
        )
        sums.append(math.fsum(np.arctan2(determinant, denominator).tolist()))
    return math.fsum(sums) / (2 * np.pi)
