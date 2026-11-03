"""Independent quadrature with strict minimum-distance triangle selection."""

import argparse
import importlib.metadata
import shutil
import time
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.reference import triangle_nearest_points
from sand_atlas_dem.util import read_json, sha256, write_json


def cull(triangles, bounds):
    # A source triangle disjoint from the target AABB contributes exactly zero.
    pad = 1e-12
    keep = np.all(
        (triangles.max(axis=1) >= bounds[0] - pad)
        & (triangles.min(axis=1) <= bounds[1] + pad),
        axis=1,
    )
    return triangles[keep]


def subdivide(triangles):
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab = (a + b) / 2
    bc = (b + c) / 2
    ca = (c + a) / 2
    return np.stack(
        [
            np.stack([a, ab, ca], axis=1),
            np.stack([ab, b, bc], axis=1),
            np.stack([ca, bc, c], axis=1),
            np.stack([ab, bc, ca], axis=1),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def enclosure_planes(vertices):
    from scipy.spatial import ConvexHull

    normals = ConvexHull(vertices).equations[:, :3]
    # Compute support against EVERY original vertex. Every original triangle is
    # a convex combination of these vertices, independently of hull offsets.
    support = np.full(len(normals), -np.inf)
    for start in range(0, len(vertices), 4096):
        support = np.maximum(
            support, (vertices[start : start + 4096] @ normals.T).max(0)
        )
    margin = 128 * np.finfo(float).eps * (1 + np.abs(vertices).max())
    return np.column_stack([normals, -support - margin])


def enclosure_mask(points, planes):
    keep = np.ones(len(points), bool)
    for start in range(0, len(planes), 64):
        ids = np.flatnonzero(keep)
        if not len(ids):
            break
        block = planes[start : start + 64]
        keep[ids] = np.all(points[ids] @ block[:, :3].T + block[:, 3] <= 0, axis=1)
    return keep


def integrate(mesh, triangles, unit, batch, planes=None):
    bary = np.full((3, 3), 1 / 6) + np.eye(3) / 2
    # Coordinates and areas here are dimensionless in the same diameter unit.
    area = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        / 2
    )
    points = np.einsum("ij,fjk->fik", bary, triangles).reshape(-1, 3)
    weights = np.repeat(area / 3, 3)
    force = np.zeros(3)
    maximum_depth = 0.0
    active = queried = 0
    for start in range(0, len(points), batch):
        p = points[start : start + batch]
        keep = np.ones(len(p), bool) if planes is None else enclosure_mask(p, planes)
        local_weights = weights[start : start + batch][keep]
        p = p[keep]
        queried += len(p)
        if not len(p):
            continue
        nearest, distance, _ = triangle_nearest_points(mesh, p)
        inside = mesh.contains(p)
        active += int(inside.sum())
        if inside.any():
            maximum_depth = max(maximum_depth, float(distance[inside].max()))
            force += (
                0.5
                * np.sum(
                    local_weights[inside, None] * (nearest[inside] - p[inside]),
                    axis=0,
                )
                * unit**3
            )
    return force, maximum_depth, active, len(points), queried


def main():
    p = argparse.ArgumentParser()
    p.add_argument("protocol", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol_hash = sha256(args.protocol)
    script_hash = sha256(__file__)
    policy = read_json(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    shutil.copy2(args.protocol, snapshot / "protocol.json")
    from sand_atlas_dem import reference as reference_module

    reference_source = Path(reference_module.__file__)
    source_hash = sha256(reference_source)
    shutil.copy2(reference_source, snapshot / "reference.py")
    if policy.get("nearest_point_policy") != "strict-fp64-minimum-squared-distance":
        raise ValueError("Explicit strict minimum query policy required")
    if source_hash != policy["reference_source_sha256"]:
        raise ValueError("Frozen reference source changed")
    library = Path(policy["library"])
    reference = Path(policy["pose_report"])
    if (
        sha256(library) != policy["library_sha256"]
        or sha256(reference) != policy["pose_report_sha256"]
    ):
        raise ValueError("Frozen source changed")
    if policy.get("outside_point_rejection") != "fp64-original-vertex-support-planes":
        raise ValueError("Explicit frozen enclosure policy required")
    shapes = {s.id: s for s in ShapeLibrary.load(library).shapes}
    rows = []
    begin = time.perf_counter()
    for case in read_json(reference)["rows"]:
        shape = shapes[case["shape"]]
        unit = (6 * shape.volume / np.pi) ** (1 / 3)
        mesh = trimesh.Trimesh(shape.vertices / unit, shape.faces, process=False)
        planes = enclosure_planes(mesh.vertices)
        branch = np.array([case["separation_m"] / unit, 0, 0])
        triangles = [
            cull(mesh.triangles - sign * branch, mesh.bounds) for sign in [1, -1]
        ]
        levels = []
        for level in policy["subdivision_levels"]:
            if level:
                triangles = [cull(subdivide(t), mesh.bounds) for t in triangles]
            force = np.zeros(3)
            depth = 0.0
            active = points = queried = 0
            for tri, sign in zip(triangles, [1, -1]):
                f, d, a, n, q = integrate(
                    mesh, tri, unit, policy["query_batch_size"], planes
                )
                force += sign * f
                depth = max(depth, d)
                active += a
                points += n
                queried += q
            levels.append(
                {
                    "level": level,
                    "force_integral_m3": force.tolist(),
                    "maximum_sampled_depth_fraction": depth,
                    "active_nodes": active,
                    "evaluated_nodes": points,
                    "triangle_queried_nodes": queried,
                    "candidate_subtriangles": [len(t) for t in triangles],
                }
            )
            print(
                shape.id, case["target_sampled_depth_fraction"], levels[-1], flush=True
            )
            write_json(
                args.output / "progress.json",
                {
                    "completed_rows": rows,
                    "current_case": case,
                    "current_levels": levels,
                },
            )
        final = np.array(levels[-1]["force_integral_m3"])
        previous = np.array(levels[-2]["force_integral_m3"])
        norm = np.linalg.norm(final)
        if norm == 0:
            raise ValueError("No integrated contact in frozen pose")
        change = float(np.linalg.norm(final - previous) / norm)
        qmc = np.array(case["exact_normal_traction_integral_m3"])
        rows.append(
            {
                "shape": shape.id,
                "separation_m": case["separation_m"],
                "target_sampled_depth_fraction": case["target_sampled_depth_fraction"],
                "levels": levels,
                "finest_relative_change": change,
                "qmc_2048_relative_error_to_finest": float(
                    np.linalg.norm(qmc - final) / norm
                ),
            }
        )
    passed = all(
        r["finest_relative_change"] <= policy["maximum_finest_relative_change"]
        for r in rows
    )
    if (
        sha256(reference_source) != source_hash
        or sha256(args.protocol) != protocol_hash
        or sha256(__file__) != script_hash
    ):
        raise ValueError("Frozen reference inputs changed during execution")
    write_json(
        args.output / "report.json",
        {
            "protocol": policy,
            "protocol_sha256": protocol_hash,
            "script_sha256": script_hash,
            "reference_source_sha256": source_hash,
            "runtime_versions": {
                package: importlib.metadata.version(package)
                for package in ("numpy", "scipy", "trimesh", "rtree")
            },
            "rows": rows,
            "passed": passed,
            "elapsed_wall_seconds": time.perf_counter() - begin,
            "scope": "Independent static normal-traction integration on original source triangles. Finest-change gate is a numerical refinement check, not a rigorous error bound or friction/material validation.",
        },
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
