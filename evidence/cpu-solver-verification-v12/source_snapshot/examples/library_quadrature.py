"""Whole-library static reference; parity first avoids exterior distance queries."""

import argparse
import importlib.metadata
import importlib.util
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.reference import triangle_nearest_points
from sand_atlas_dem.util import read_json, sha256, write_json

helper_path = Path(__file__).with_name("triangle_quadrature_minimum.py")
helper_spec = importlib.util.spec_from_file_location(
    "strict_quadrature_geometry", helper_path
)
geometry = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(geometry)


def integrate(mesh, triangles, unit, batch, planes=None):
    """Same Gauss3 integral; only interior nodes need a nearest-point query."""
    bary = np.full((3, 3), 1 / 6) + np.eye(3) / 2
    areas = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        / 2
    )
    points = np.einsum("ij,fjk->fik", bary, triangles).reshape(-1, 3)
    weights = np.repeat(areas / 3, 3)
    force = np.zeros(3)
    depth = 0.0
    active = queried = 0
    for start in range(0, len(points), batch):
        p = points[start : start + batch]
        keep = (
            np.ones(len(p), bool)
            if planes is None
            else geometry.enclosure_mask(p, planes)
        )
        local_weights, p = weights[start : start + batch][keep], p[keep]
        if not len(p):
            continue
        inside = mesh.contains(p)
        active += int(inside.sum())
        if not inside.any():
            continue
        interior = p[inside]
        nearest, distance, _ = triangle_nearest_points(mesh, interior)
        queried += len(interior)
        depth = max(depth, float(distance.max()))
        force += (
            0.5
            * np.sum(local_weights[inside, None] * (nearest - interior), axis=0)
            * unit**3
        )
    # Fifth field now counts actual nearest-point queries, not parity candidates.
    return force, depth, active, len(points), queried


def study_case(shape, case, policy, progress_path):
    unit = (6 * shape.volume / np.pi) ** (1 / 3)
    mesh = trimesh.Trimesh(shape.vertices / unit, shape.faces, process=False)
    planes = geometry.enclosure_planes(mesh.vertices)
    branch = np.array([case["separation_m"] / unit, 0, 0])
    triangles = [
        geometry.cull(mesh.triangles - sign * branch, mesh.bounds) for sign in (1, -1)
    ]
    levels = []
    for level in policy["subdivision_levels"]:
        if level:
            triangles = [
                geometry.cull(geometry.subdivide(t), mesh.bounds) for t in triangles
            ]
        force = np.zeros(3)
        depth = 0.0
        active = points = queried = 0
        for tri, sign in zip(triangles, (1, -1)):
            f, d, a, n, q = integrate(
                mesh, tri, unit, policy["query_batch_size"], planes
            )
            force += sign * f
            depth = max(depth, d)
            active += a
            points += n
            queried += q
        levels.append(
            dict(
                level=level,
                force_integral_m3=force,
                maximum_sampled_depth_fraction=depth,
                active_nodes=active,
                evaluated_nodes=points,
                triangle_queried_nodes=queried,
                candidate_subtriangles=[len(t) for t in triangles],
            )
        )
        write_json(progress_path, dict(case=case, levels=levels))
    final = np.array(levels[-1]["force_integral_m3"])
    previous = np.array(levels[-2]["force_integral_m3"])
    norm = np.linalg.norm(final)
    if not np.isfinite(final).all() or norm == 0:
        raise ValueError("No finite nonzero integrated contact in frozen pose")
    return dict(
        shape=shape.id,
        separation_m=case["separation_m"],
        target_sampled_depth_fraction=case["target_sampled_depth_fraction"],
        levels=levels,
        finest_relative_change=float(np.linalg.norm(final - previous) / norm),
        qmc_2048_relative_error_to_finest=float(
            np.linalg.norm(np.array(case["exact_normal_traction_integral_m3"]) - final)
            / norm
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = read_json(args.protocol)
    protocol_hash = sha256(args.protocol)
    for name in ("library", "pose_report"):
        if sha256(policy[name]) != policy[name + "_sha256"]:
            raise ValueError(f"Frozen {name} changed")
    if (
        policy["nearest_point_policy"] != "strict-fp64-minimum-squared-distance"
        or policy["query_order"] != "parity-before-interior-nearest"
        or policy["outside_point_rejection"] != "fp64-original-vertex-support-planes"
        or policy["subdivision_levels"] != list(range(5))
    ):
        raise ValueError("Unsupported frozen reference contract")
    poses = read_json(policy["pose_report"])
    if not poses["passed"]:
        raise ValueError("Source pose preparation failed")
    library = ShapeLibrary.load(policy["library"])
    shapes = {s.id: s for s in library.shapes}
    cases = poses["rows"]
    keys = {(c["shape"], c["target_sampled_depth_fraction"]) for c in cases}
    expected = {
        (s, t) for s in shapes for t in policy["target_sampled_depth_fractions"]
    }
    if (
        keys != expected
        or len(keys) != len(cases)
        or sorted(shapes) != policy["shape_ids"]
    ):
        raise ValueError(
            "Incomplete, duplicated or changed whole-library pose inventory"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    private = args.output / "private"
    private.mkdir(mode=0o700)
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        helper_path,
        *sorted((root / "src/sand_atlas_dem").glob("*.py")),
    ]
    frozen = {str(p.relative_to(root)): sha256(p) for p in paths}
    for path in paths:
        target = args.output / "source_snapshot" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(args.protocol, args.output / "protocol.json")
    write_json(
        args.output / "prospective-freeze.json",
        dict(source_hashes=frozen, protocol_sha256=protocol_hash),
    )
    rows, failures = [], []
    begin = time.perf_counter()
    with ProcessPoolExecutor(max_workers=policy["cpu_workers"]) as pool:
        jobs = {}
        for case in cases:
            progress = private / (
                case["shape"]
                + "-"
                + str(case["target_sampled_depth_fraction"])
                + ".json"
            )
            job = pool.submit(study_case, shapes[case["shape"]], case, policy, progress)
            jobs[job] = (case, progress)
        for job in as_completed(jobs):
            case, progress = jobs[job]
            try:
                row = job.result()
                rows.append(row)
                print(
                    row["shape"],
                    row["target_sampled_depth_fraction"],
                    row["finest_relative_change"],
                    flush=True,
                )
            except Exception as exc:
                failure = dict(
                    shape=case["shape"],
                    separation_m=case["separation_m"],
                    error=f"{type(exc).__name__}: {exc}",
                    partial=read_json(progress) if progress.exists() else None,
                )
                failures.append(failure)
                print(failure, flush=True)
            write_json(
                args.output / "progress.json", dict(rows=rows, failures=failures)
            )
    if sha256(args.protocol) != protocol_hash or any(
        sha256(root / p) != h for p, h in frozen.items()
    ):
        raise ValueError("Frozen source or protocol changed during execution")
    rows.sort(key=lambda r: (r["shape"], r["target_sampled_depth_fraction"]))
    passed = (
        not failures
        and len(rows) == len(cases)
        and all(
            r["finest_relative_change"] <= policy["maximum_finest_relative_change"]
            for r in rows
        )
    )
    write_json(
        args.output / "report.json",
        dict(
            protocol=policy,
            protocol_sha256=protocol_hash,
            source_hashes=frozen,
            rows=rows,
            failures=failures,
            passed=passed,
            elapsed_wall_seconds=time.perf_counter() - begin,
            runtime_versions={
                p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "trimesh", "rtree")
            },
            scope="Independent static source-triangle Gauss3 refinement at every frozen source-library pose. The unchanged 2% finest-change gate is a refinement check, not a rigorous absolute error bound or material validation.",
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
