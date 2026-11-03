"""Freeze accepted static evidence and policy before a bounded Metal preparation."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    policy = read_json(args.policy)
    outer_path = Path(policy["requires_qualification"])
    gpu_path = Path(policy["requires_static_gpu_report"])
    outer, gpu = read_json(outer_path), read_json(gpu_path)
    expected = policy["expected_gpu_cases"]
    if not (
        outer.get("passed")
        and outer.get("source_unchanged")
        and gpu.get("passed")
        and gpu.get("complete")
        and outer["completed_gpu_cases"] == expected
        and outer["expected_gpu_cases"] == expected
        and len(gpu["rows"]) == expected
    ):
        raise ValueError("Complete accepted static GPU qualification required")
    experiment = Path(policy["experiment"])
    spec = Experiment.load(experiment)
    candidate = Path(spec.shape_library)
    fine = gpu["libraries"][-1]
    if (
        candidate.resolve() != Path(fine["path"]).resolve()
        or sha256(candidate) != fine["sha256"]
    ):
        raise ValueError("Experiment does not name the qualified candidate bytes")
    ref_path = Path(gpu["protocol"]["reference_report"])
    reference = read_json(ref_path)
    if sha256(ref_path) != gpu["reference_report_sha256"] or not reference["passed"]:
        raise ValueError("Changed independent reference")
    expected_poses = {(r["shape"], r["separation_m"]) for r in reference["rows"]}
    if len(expected_poses) != policy["expected_reference_cases"]:
        raise ValueError("Reference inventory differs from preparation policy")
    for index in (0, 1):
        rows = [r for r in gpu["rows"] if r["library_index"] == index]
        if (
            len(rows) != len(expected_poses)
            or {(r["shape"], r["separation_m"]) for r in rows} != expected_poses
        ):
            raise ValueError("Incomplete unique GPU pose inventory")
    if (
        spec.particles != policy["expected_particles"]
        or spec.shape_sampling != "balanced"
        or spec.contact_geometry != policy["expected_contact_geometry"]
    ):
        raise ValueError("Experiment inventory or geometry differs from policy")
    root = Path(__file__).resolve().parents[1]
    for name, digest in gpu["source_hashes"].items():
        if sha256(root / name) != digest:
            raise ValueError("Preparation solver differs from qualified static solver")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        args.policy.resolve(),
        experiment.resolve(),
        outer_path.resolve(),
        gpu_path.resolve(),
        ref_path.resolve(),
        Path(__file__).resolve(),
        root / "examples/frozen_controller.py",
        root / "examples/audit_preparation.py",
    ]
    inputs = {str(p): sha256(p) for p in paths}
    for i, path in enumerate(paths):
        shutil.copy2(path, output / (str(i) + "-" + path.name))
    write_json(
        output / "prospective-freeze.json",
        dict(
            identity=policy["identity"],
            inputs=inputs,
            candidate_sha256=fine["sha256"],
            policy=policy,
            expected_source_shapes=policy["expected_shapes"],
            scientific_admission=False,
            status="frozen-before-preparation",
        ),
    )
    with (output / "preparation.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(root / "examples/frozen_controller.py"),
                str(experiment),
                "--output",
                str(output / "study"),
                "--chunks-per-checkpoint",
                str(policy["chunks_per_checkpoint"]),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    unchanged = all(sha256(path) == digest for path, digest in inputs.items())
    # Terminal controller success is still subject to the independent state audit.
    audit_code = None
    if result.returncode == 0 and unchanged:
        audit_code = subprocess.run(
            [
                sys.executable,
                str(root / "examples/audit_preparation.py"),
                str(output / "study"),
                "--policy",
                str(args.policy),
                "--output",
                str(output / "acceptance-audit.json"),
            ]
        ).returncode
    write_json(
        output / "report.json",
        dict(
            identity=policy["identity"],
            passed=result.returncode == 0 and unchanged and audit_code == 0,
            controller_returncode=result.returncode,
            audit_returncode=audit_code,
            source_unchanged=unchanged,
            scientific_admission=False,
            scope="Bounded balanced mesh preparation only; not dynamic convergence, 10k pilot or material validation.",
        ),
    )
    return 0 if result.returncode == 0 and unchanged and audit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
