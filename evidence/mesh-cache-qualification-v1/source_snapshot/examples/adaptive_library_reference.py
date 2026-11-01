"""Prospective whole-library reference rerun in an immutable isolated source copy.

Every original pose runs levels 0..4. A failed 2% finest-change gate triggers
levels 5 and 6, without deleting the original levels or changing the tolerance.
This qualifies only the reference integrals, not a production quadrature asset.
"""

import argparse
import importlib.util
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.util import read_json, sha256, write_json

source = Path(__file__).with_name("library_quadrature_deterministic.py")
module_spec = importlib.util.spec_from_file_location(
    "adaptive_reference_operator", source
)
reference = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(reference)


def evaluate(shape, case, policy, progress):
    initial = reference.study_case(shape, case, policy, progress)
    result = initial
    if initial["finest_relative_change"] > policy["maximum_finest_relative_change"]:
        extended = dict(policy, subdivision_levels=list(range(7)))
        result = reference.study_case(shape, case, extended, progress)
        for a, b in zip(initial["levels"], result["levels"]):
            if not np.array_equal(a["force_integral_m3"], b["force_integral_m3"]):
                raise ValueError(
                    "Adaptive extension changed an overlapping reference level"
                )
    result["level4_relative_change"] = initial["finest_relative_change"]
    result["reference_finest_level"] = result["levels"][-1]["level"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = read_json(args.protocol)
    if (
        policy["subdivision_levels"] != list(range(5))
        or policy["adaptive_extension_levels"] != [5, 6]
        or policy["maximum_finest_relative_change"] != 0.02
        or policy["reference_source_sha256"] != sha256(source)
    ):
        raise ValueError("Unsupported frozen adaptive reference contract")
    for name in ("library", "pose_report"):
        if sha256(policy[name]) != policy[name + "_sha256"]:
            raise ValueError(f"Changed frozen {name}")
    poses = read_json(policy["pose_report"])
    library = ShapeLibrary.load(policy["library"])
    shapes = {s.id: s for s in library.shapes}
    cases = poses["rows"]
    expected = {
        (s, d) for s in shapes for d in policy["target_sampled_depth_fractions"]
    }
    keys = {(c["shape"], c["target_sampled_depth_fraction"]) for c in cases}
    if (
        not poses["passed"]
        or keys != expected
        or len(cases) != len(expected)
        or sorted(shapes) != policy["shape_ids"]
    ):
        raise ValueError("Incomplete, changed or duplicated original pose inventory")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "private").mkdir(mode=0o700)
    root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        source,
        reference.helper_path,
        reference.ray_path,
        *reference.ray_dependencies,
        *sorted((root / "src/sand_atlas_dem").glob("*.py")),
    ]
    hashes = {str(p.relative_to(root)): sha256(p) for p in files}
    for path in files:
        target = output / "source_snapshot" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    protocol_hash = sha256(args.protocol)
    shutil.copy2(args.protocol, output / "protocol.json")
    write_json(
        output / "prospective-freeze.json",
        dict(source_hashes=hashes, protocol_sha256=protocol_hash),
    )
    begin, rows, failures = time.perf_counter(), [], []
    with ProcessPoolExecutor(max_workers=policy["cpu_workers"]) as pool:
        jobs = {}
        for case in cases:
            progress = (
                output
                / "private"
                / (
                    case["shape"]
                    + "-"
                    + str(case["target_sampled_depth_fraction"])
                    + ".json"
                )
            )
            jobs[
                pool.submit(evaluate, shapes[case["shape"]], case, policy, progress)
            ] = case
        for job in as_completed(jobs):
            case = jobs[job]
            try:
                row = job.result()
                rows.append(row)
                print(
                    row["shape"],
                    row["target_sampled_depth_fraction"],
                    row["reference_finest_level"],
                    row["finest_relative_change"],
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    dict(
                        shape=case["shape"],
                        depth=case["target_sampled_depth_fraction"],
                        error=repr(exc),
                    )
                )
            write_json(output / "progress.json", dict(rows=rows, failures=failures))
    unchanged = (
        all(sha256(root / name) == digest for name, digest in hashes.items())
        and sha256(args.protocol) == protocol_hash
    )
    refined = bool(
        len(rows) == len(cases)
        and all(
            r["finest_relative_change"] <= policy["maximum_finest_relative_change"]
            for r in rows
        )
    )
    passed = bool(unchanged and not failures and refined)
    write_json(
        output / "report.json",
        dict(
            identity=policy["identity"],
            passed=passed,
            source_unchanged=unchanged,
            reference_refinement_passed=refined,
            expected_cases=len(cases),
            completed_cases=len(rows),
            failures=failures,
            rows=sorted(
                rows, key=lambda r: (r["shape"], r["target_sampled_depth_fraction"])
            ),
            elapsed_seconds=time.perf_counter() - begin,
            production_quadrature_asset_admitted=False,
            whole_library_qualified=False,
            scientific_admission=False,
            scope=__doc__,
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
