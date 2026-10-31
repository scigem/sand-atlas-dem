"""Diagnose the retained GPU/reference failures without changing their gates."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.reference import triangle_nearest_points
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = Path("evidence/hostun-gpu-triangle-quadrature-v1/report.json")
    failure = read_json(parent)
    policy = failure["protocol"]
    reference_path = Path(policy["reference_report"])
    if sha256(reference_path) != policy["reference_report_sha256"]:
        raise ValueError("Reference changed")
    reference = read_json(reference_path)
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        *sorted((root / "src/sand_atlas_dem").glob("*.py")),
        *sorted((root / "src/sand_atlas_dem/shaders").glob("*.slang")),
    ]
    frozen = {str(p.relative_to(root)): sha256(p) for p in paths}
    for path in paths:
        target = args.output / "source_snapshot" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    freeze = dict(
        parent_report_sha256=sha256(parent),
        source_hashes=frozen,
        policy=policy,
        scope="Observational diagnosis, no acceptance gate.",
        exhaustive_check_nodes_per_case=12,
    )
    write_json(args.output / "prospective-freeze.json", freeze)
    metal = Metal()
    rows = []
    for asset in policy["libraries"]:
        if sha256(asset["path"]) != asset["sha256"]:
            raise ValueError("Asset changed")
        library = ShapeLibrary.load(asset["path"])
        failures = [
            r
            for r in failure["rows"]
            if r["subdivisions"] == asset["subdivisions"]
            and r["fixed_level_gpu_relative_error"]
            > policy["maximum_fixed_level_gpu_error"]
        ]
        for failed in failures:
            shape = next(s for s in library.shapes if s.id == failed["shape"])
            case = next(
                r
                for r in reference["rows"]
                if r["shape"] == shape.id
                and r["separation_m"] == failed["separation_m"]
            )
            origin = np.full(3, 4 * shape.radius)
            spec = Experiment(
                particles=2,
                timestep=policy["timestep_s"],
                contact_geometry="mesh",
                friction=0,
                max_overlap_fraction=policy["overlap_guard_fraction"],
                contacts_per_particle=policy["contacts_per_particle"],
                pairs_per_particle=4,
            )
            engine = Engine(
                ShapeLibrary([shape]),
                spec,
                [origin, origin + [case["separation_m"], 0, 0]],
                np.full(3, 20 * shape.radius),
                [0, 0],
                [1.0, 1.0],
                metal=metal,
            )
            engine.initialize()
            count = int(metal.read(engine.buffers["control"], np.uint32)[2])
            committed = engine.history_buffers[1 - engine.active]
            meta = metal.read(committed["meta"], np.uint32).reshape(-1, 4)[:count]
            pd = metal.read(engine.buffers["pointDepth"]).reshape(-1, 4)[:count]
            na = (
                metal.read(committed["normalArea"]).reshape(-1, 4)[:count].astype(float)
            )
            pos = metal.read(engine.buffers["position"]).reshape(-1, 4)
            rem = metal.read(engine.buffers["positionRemainder"]).reshape(-1, 4)
            branch = (pos[1, :3] - pos[0, :3]) + (rem[1, :3] - rem[0, :3])
            signs = np.where(meta[:, 0] == 0, np.float32(1), np.float32(-1))
            points = (pd[:, :3] - signs[:, None] * branch).astype(float)
            mesh = trimesh.Trimesh(
                shape.vertices / engine.length_unit, shape.faces, process=False
            )
            strict, _, _ = triangle_nearest_points(mesh, points)
            legacy, _, _ = trimesh.proximity.closest_point(mesh, points)
            gpu_delta = pd[:, 3, None].astype(float) * na[:, :3]
            strict_delta = strict - points
            legacy_delta = legacy - points
            total = lambda delta: np.sum(
                0.5 * signs[:, None] * na[:, 3, None] * delta, axis=0
            )
            error = np.linalg.norm(gpu_delta - legacy_delta, axis=1)
            nodes = np.argsort(error)[-freeze["exhaustive_check_nodes_per_case"] :]
            exhaustive_errors = []
            for node in nodes:
                exhaustive = trimesh.triangles.closest_point(
                    mesh.triangles, np.broadcast_to(points[node], (len(mesh.faces), 3))
                )
                squared = np.sum((exhaustive - points[node]) ** 2, axis=1)
                exhaustive_errors.append(
                    abs(float(np.min(squared) - np.sum(strict_delta[node] ** 2)))
                )
            row = dict(
                shape=shape.id,
                subdivisions=asset["subdivisions"],
                active_nodes=count,
                maximum_gpu_displacement_error_diameters=float(
                    np.linalg.norm(gpu_delta - strict_delta, axis=1).max()
                ),
                strict_minimum_gpu_relative_force_error=float(
                    np.linalg.norm(total(gpu_delta) - total(strict_delta))
                    / np.linalg.norm(total(strict_delta))
                ),
                maximum_legacy_excess_squared_distance=float(
                    np.max(
                        np.sum(legacy_delta**2, axis=1)
                        - np.sum(strict_delta**2, axis=1)
                    )
                ),
                exhaustive_checked_nodes=len(nodes),
                maximum_candidate_vs_exhaustive_squared_distance_error=max(
                    exhaustive_errors
                ),
            )
            rows.append(row)
            write_json(args.output / "progress.json", rows)
            print(row, flush=True)
            del engine
        del library
    if (
        any(sha256(root / name) != digest for name, digest in frozen.items())
        or sha256(parent) != freeze["parent_report_sha256"]
    ):
        raise ValueError("Executed source or failed parent report changed")
    write_json(
        args.output / "report.json",
        dict(
            freeze=freeze,
            rows=rows,
            scope="Only retained active nodes; does not check missing contacts or qualify the full force integral. No acceptance claim.",
        ),
    )


if __name__ == "__main__":
    main()
