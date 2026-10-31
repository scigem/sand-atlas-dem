"""Prospectively frozen higher-resolution diagnostic for one failed source pose."""

import argparse
import importlib.util
import shutil
from pathlib import Path

import numpy as np

from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.util import read_json, sha256, write_json

source = Path(__file__).with_name("library_quadrature_deterministic.py")
module_spec = importlib.util.spec_from_file_location("deterministic_case", source)
reference = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(reference)


def overlap_difference(previous, current):
    """Exact zero stays zero; relative differences require a nonzero denominator.

    None denotes a failed zero-reference overlap, never a relaxed tolerance.
    """
    a, b = np.asarray(previous, float), np.asarray(current, float)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
        raise ValueError("Finite three-vector overlap forces required")
    norm = np.linalg.norm(a)
    if norm == 0:
        return 0.0 if np.array_equal(a, b) else None
    return float(np.linalg.norm(a - b) / norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = read_json(args.policy)
    for name in ("library", "pose_report"):
        if sha256(policy[name]) != policy[name + "_sha256"]:
            raise ValueError(f"Changed frozen {name}")
    if sha256(source) != policy["reference_source_sha256"]:
        raise ValueError("Changed reference operator")
    if (
        policy["subdivision_levels"] != list(range(7))
        or policy["maximum_finest_relative_change"] != 0.02
        or policy["maximum_overlap_relative_difference"] != 1e-12
    ):
        raise ValueError("Unsupported targeted refinement contract")
    poses = read_json(policy["pose_report"])
    case = [
        r
        for r in poses["rows"]
        if r["shape"] == policy["shape"]
        and r["target_sampled_depth_fraction"]
        == policy["target_sampled_depth_fraction"]
    ]
    if len(case) != 1 or not poses["passed"]:
        raise ValueError("Missing unique accepted source pose")
    library = ShapeLibrary.load(policy["library"])
    shape = next(s for s in library.shapes if s.id == policy["shape"])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "private").mkdir(mode=0o700)
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
    for p in files:
        dest = args.output / "source_snapshot" / p.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    shutil.copy2(args.policy, args.output / "protocol.json")
    write_json(
        args.output / "prospective-freeze.json",
        dict(source_hashes=hashes, protocol_sha256=sha256(args.policy)),
    )
    row = reference.study_case(
        shape, case[0], policy, args.output / "private/levels.json"
    )
    differences = []
    for previous, current in zip(policy["baseline_row"]["levels"], row["levels"]):
        if previous["level"] != current["level"]:
            raise ValueError("Changed level identity")
        a, b = (
            np.array(previous["force_integral_m3"]),
            np.array(current["force_integral_m3"]),
        )
        differences.append(overlap_difference(a, b))
    unchanged = all(sha256(root / p) == h for p, h in hashes.items()) and sha256(
        args.policy
    ) == sha256(args.output / "protocol.json")
    overlap_passed = bool(differences) and all(
        d is not None and d <= policy["maximum_overlap_relative_difference"]
        for d in differences
    )
    refinement_passed = (
        row["finest_relative_change"] <= policy["maximum_finest_relative_change"]
    )
    write_json(
        args.output / "report.json",
        dict(
            identity=policy["identity"],
            passed=bool(unchanged and overlap_passed and refinement_passed),
            source_unchanged=unchanged,
            common_level_relative_differences=differences,
            overlap_passed=bool(overlap_passed),
            refinement_passed=bool(refinement_passed),
            row=row,
            whole_library_qualified=False,
            scientific_admission=False,
        ),
    )
    return 0 if unchanged and overlap_passed and refinement_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
