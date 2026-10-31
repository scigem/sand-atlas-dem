"""CPU source-derived contact poses for every grain, without geometry repair."""

import argparse
import importlib.metadata
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.reference import triangle_nearest_points
from sand_atlas_dem.util import read_json, sha256, write_json


def evaluate(shape, mesh, separation, batch):
    unit = (6 * shape.volume / np.pi) ** (1 / 3)
    maximum = 0.0
    force = np.zeros(3)
    counts = []
    for sign in (1, -1):
        points = shape.samples / unit - sign * np.array([separation, 0, 0])
        mask = np.all((points >= mesh.bounds[0]) & (points <= mesh.bounds[1]), axis=1)
        points, areas = points[mask], shape.areas[mask]
        count = 0
        for start in range(0, len(points), batch):
            block = points[start : start + batch]
            inside = mesh.contains(block)
            count += int(inside.sum())
            if not inside.any():
                continue
            active = block[inside]
            nearest, distance, _ = triangle_nearest_points(mesh, active)
            maximum = max(maximum, float(distance.max()))
            force += (
                0.5
                * sign
                * unit
                * np.sum(
                    areas[start : start + len(block)][inside, None]
                    * (nearest - active),
                    axis=0,
                )
            )
        counts.append(count)
    return maximum, force, counts


def prepare_shape(shape, policy, progress_path=None):
    unit = (6 * shape.volume / np.pi) ** (1 / 3)
    mesh = trimesh.Trimesh(shape.vertices / unit, shape.faces, process=False)
    targets = policy["target_sampled_depth_fractions"]
    brackets = {}
    rows = []

    def retain(**state):
        if progress_path is not None:
            write_json(
                progress_path,
                dict(shape=shape.id, brackets=brackets, rows=rows, **state),
            )

    previous = 2.1 * shape.radius / unit
    previous_depth, _, _ = evaluate(shape, mesh, previous, policy["query_batch_size"])
    if previous_depth != 0:
        raise ValueError("Bounding-sphere separation was not exterior")
    scan = np.linspace(previous, 0, policy["bracket_scan_intervals"] + 1)[1:]
    for separation in scan:
        depth, _, _ = evaluate(shape, mesh, separation, policy["query_batch_size"])
        for target in targets:
            if target not in brackets and previous_depth < target <= depth:
                brackets[target] = [float(separation), previous]
                retain(
                    phase="bracketing",
                    last_separation_diameters=float(separation),
                    last_depth=depth,
                )
        if len(brackets) == len(targets):
            break
        previous, previous_depth = float(separation), depth
    if len(brackets) != len(targets):
        raise ValueError(f"No source-node depth bracket for {shape.id}")
    for target in targets:
        lower, upper = brackets[target]
        initial = [lower, upper]
        for _ in range(policy["bisection_iterations"]):
            middle = (lower + upper) / 2
            depth, _, _ = evaluate(shape, mesh, middle, policy["query_batch_size"])
            if depth >= target:
                lower = middle
            else:
                upper = middle
            retain(
                phase="bisection",
                target=target,
                lower=lower,
                upper=upper,
                last_separation_diameters=middle,
                last_depth=depth,
            )
        depth, force, counts = evaluate(shape, mesh, upper, policy["query_batch_size"])
        if (
            depth > target
            or target - depth > policy["maximum_depth_shortfall_fraction"]
            or upper - lower > policy["maximum_bracket_width_fraction"]
            or not np.isfinite(force).all()
            or np.linalg.norm(force) == 0
        ):
            raise ValueError(
                f"Independent pose acceptance failed for {shape.id}: {target}, {depth}"
            )
        rows.append(
            dict(
                shape=shape.id,
                source_sha256=shape.provenance["source_sha256"],
                target_sampled_depth_fraction=target,
                separation_m=upper * unit,
                maximum_sampled_depth_fraction=depth,
                exact_normal_traction_integral_m3=force,
                exact_active_nodes=counts,
                sample_count=len(shape.samples),
                initial_bracket_diameters=initial,
                final_bracket_diameters=[lower, upper],
            )
        )
        retain(phase="target_complete", target=target)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = read_json(args.protocol)
    protocol_hash = sha256(args.protocol)
    if policy["nearest_point_policy"] != "strict-fp64-minimum-squared-distance":
        raise ValueError("Explicit strict-distance query policy required")
    if sha256(policy["library"]) != policy["library_sha256"]:
        raise ValueError("Frozen library changed")
    library = ShapeLibrary.load(policy["library"])
    if sorted(s.id for s in library.shapes) != policy["shape_ids"]:
        raise ValueError("Frozen whole-library inventory changed")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), *sorted((root / "src/sand_atlas_dem").glob("*.py"))]
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
    private_progress = args.output / "private" / "grain-progress"
    private_progress.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=policy["cpu_workers"]) as pool:
        jobs = {
            pool.submit(
                prepare_shape, shape, policy, private_progress / (shape.id + ".json")
            ): shape.id
            for shape in library.shapes
        }
        for job in as_completed(jobs):
            try:
                completed = job.result()
                rows.extend(completed)
                print(
                    jobs[job],
                    "complete",
                    [
                        (r["target_sampled_depth_fraction"], r["exact_active_nodes"])
                        for r in completed
                    ],
                    flush=True,
                )
            except Exception as exc:
                failure = dict(shape=jobs[job], error=f"{type(exc).__name__}: {exc}")
                failures.append(failure)
                print(failure, flush=True)
            write_json(
                args.output / "progress.json", dict(rows=rows, failures=failures)
            )
    if sha256(args.protocol) != protocol_hash or any(
        sha256(root / p) != h for p, h in frozen.items()
    ):
        raise ValueError("Executed source or protocol changed")
    rows.sort(key=lambda r: (r["shape"], r["target_sampled_depth_fraction"]))
    passed = not failures and len(rows) == len(library.shapes) * len(
        policy["target_sampled_depth_fractions"]
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
            runtime_versions={
                p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "trimesh", "rtree")
            },
            scope="Frozen original-node poses and strict FP64 force references for the whole accepted source library. No refined surface-quadrature, dynamic, friction, assembly or material qualification.",
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
