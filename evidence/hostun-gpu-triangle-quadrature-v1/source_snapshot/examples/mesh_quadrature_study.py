"""Frozen GPU quadrature refinement against an independent source-triangle integral."""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_hash = sha256(args.protocol)
    policy = read_json(args.protocol)
    reference_path = Path(policy["reference_report"])
    if sha256(reference_path) != policy["reference_report_sha256"]:
        raise ValueError("Reference bytes changed")
    reference = read_json(reference_path)
    if (
        not reference["passed"]
        or max(r["finest_relative_change"] for r in reference["rows"]) > 0.02
    ):
        raise ValueError("Independent reference failed its unchanged refinement gate")
    for item in policy["libraries"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("Prepared asset bytes changed")
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
    shutil.copy2(args.protocol, args.output / "protocol.json")
    write_json(
        args.output / "prospective-freeze.json",
        dict(protocol_sha256=sha256(args.protocol), files=frozen),
    )
    metal = Metal()
    rows = []
    library_records = []
    for item in policy["libraries"]:
        library = ShapeLibrary.load(item["path"])
        if any(
            s.provenance.get("parent_shape_library_sha256")
            != policy["parent_library_sha256"]
            for s in library.shapes
        ):
            raise ValueError(
                "Quadrature asset does not name the frozen physical source library"
            )
        shapes = {s.id: s for s in library.shapes}
        library_records.append(
            dict(
                **item,
                shape_count=len(shapes),
                sample_count=sum(len(s.samples) for s in library.shapes),
            )
        )
        for case in reference["rows"]:
            shape = shapes[case["shape"]]
            row = dict(
                subdivisions=item["subdivisions"],
                shape=shape.id,
                separation_m=case["separation_m"],
                target_sampled_depth_fraction=case["target_sampled_depth_fraction"],
            )
            try:
                origin = np.full(3, 4 * shape.radius)
                positions = np.array([origin, origin + [case["separation_m"], 0, 0]])
                spec = Experiment(
                    particles=2,
                    timestep=policy["timestep_s"],
                    contact_geometry="mesh",
                    friction=0,
                    max_overlap_fraction=policy["overlap_guard_fraction"],
                    contacts_per_particle=policy["contacts_per_particle"],
                    pairs_per_particle=4,
                )
                begin = time.perf_counter()
                engine = Engine(
                    ShapeLibrary([shape]),
                    spec,
                    positions,
                    np.full(3, 20 * shape.radius),
                    [0, 0],
                    [1.0, 1.0],
                    metal=metal,
                )
                engine.initialize()
                force = (
                    metal.read(engine.buffers["force"]).reshape(-1, 4)[0, :3]
                    * engine.force_unit
                )
                expected = (
                    np.array(case["levels"][-1]["force_integral_m3"])
                    * spec.normal_stiffness
                )
                same_level = next(
                    l for l in case["levels"] if l["level"] == item["subdivisions"]
                )
                fixed = (
                    np.array(same_level["force_integral_m3"]) * spec.normal_stiffness
                )
                diag = engine.diagnostics()
                row.update(
                    status="complete",
                    force_N=force,
                    reference_N=expected,
                    relative_force_error=float(
                        np.linalg.norm(force - expected) / np.linalg.norm(expected)
                    ),
                    fixed_level_gpu_relative_error=float(
                        np.linalg.norm(force - fixed) / np.linalg.norm(fixed)
                    ),
                    active_contacts=diag["active_surface_contacts"],
                    fp64_reference_active_nodes=same_level["active_nodes"],
                    max_overlap_fraction=diag["max_overlap_fraction"],
                    sample_count=len(shape.samples),
                    estimated_working_bytes=engine.estimated_bytes,
                    setup_and_initialization_wall_seconds=time.perf_counter() - begin,
                )
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            write_json(
                args.output / "progress.json",
                dict(libraries=library_records, rows=rows),
            )
            print(row, flush=True)
        del shapes, library
    coarse = [
        r for r in rows if r["subdivisions"] == policy["libraries"][0]["subdivisions"]
    ]
    fine = [
        r for r in rows if r["subdivisions"] == policy["libraries"][-1]["subdivisions"]
    ]
    complete = all(r["status"] == "complete" for r in rows)
    coarse_worst = max(
        (r.get("relative_force_error", float("inf")) for r in coarse),
        default=float("inf"),
    )
    fine_worst = max(
        (r.get("relative_force_error", float("inf")) for r in fine),
        default=float("inf"),
    )
    passed = (
        complete
        and fine_worst <= policy["maximum_relative_force_error"]
        and fine_worst * policy["minimum_worst_error_improvement_factor"]
        <= coarse_worst
        and all(
            row["fixed_level_gpu_relative_error"]
            <= policy["maximum_fixed_level_gpu_error"]
            for row in rows
        )
    )
    if any(sha256(root / name) != digest for name, digest in frozen.items()):
        raise ValueError("Executed solver source changed during the study")
    if (
        sha256(args.protocol) != protocol_hash
        or sha256(reference_path) != policy["reference_report_sha256"]
    ):
        raise ValueError("Frozen protocol/reference changed during the study")
    write_json(
        args.output / "report.json",
        dict(
            protocol=policy,
            protocol_sha256=sha256(args.protocol),
            reference_report_sha256=sha256(reference_path),
            source_hashes=frozen,
            backend=metal.doctor(),
            libraries=library_records,
            rows=rows,
            passed=passed,
            complete=complete,
            coarse_worst_relative_error=coarse_worst
            if np.isfinite(coarse_worst)
            else None,
            fine_worst_relative_error=fine_worst if np.isfinite(fine_worst) else None,
            scope="Ten frozen static source poses; comparison to independently refined normal-traction integral. FP64/FP32 active counts are diagnostic; the separate unchanged fixed-node force/count oracle remains required. No dynamic/friction/assembly/material or sustained throughput qualification.",
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
