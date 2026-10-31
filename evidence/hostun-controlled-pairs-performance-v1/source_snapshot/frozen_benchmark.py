"""Freeze source/input bytes and run the declared performance workloads serially."""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sand_atlas_dem.util import read_json, sha256, write_json


def execute(protocol, output):
    from sand_atlas_dem.benchmark import benchmark

    study = read_json(protocol)
    library = Path(study["library"])
    if sha256(library) != study["library_sha256"]:
        raise ValueError("Frozen library hash mismatch")
    # These settings are fixed by the implementation; reject unsupported requests.
    if (
        study["packing"] != "controlled-pairs"
        or study["orientation_seed"] != 123
        or study["target_maximum_penetration_fraction_of_equivalent_diameter"] != 0.0025
        or study["reset_to_identical_committed_state_before_each_repeat"] is not True
        or study["reset_and_diagnostic_time_in_per_step_timing"] is not False
    ):
        raise ValueError("Unsupported workload contract")
    for particles in study["particles"]:
        benchmark(
            output / f"particles-{particles}",
            particles=particles,
            library_path=library,
            packing=study["packing"],
            timestep=study["timestep_s"],
            steps=study["steps_per_repeat"],
            repeats=study["minimum_repeats"],
            minimum_seconds=study["minimum_wall_seconds"],
            pairs_per_particle=study["pair_capacity_per_particle"],
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("protocol", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute-frozen", action="store_true")
    args = p.parse_args()
    if args.execute_frozen:
        execute(args.protocol, args.output)
        return 0
    root = Path(__file__).resolve().parents[1]
    study = read_json(args.protocol)
    source_library = root / study["library"]
    if sha256(source_library) != study["library_sha256"]:
        raise ValueError("Source library hash mismatch")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    shutil.copytree(
        root / "src/sand_atlas_dem",
        snapshot / "sand_atlas_dem",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(__file__, snapshot / "frozen_benchmark.py")
    for name in ["pyproject.toml", "requirements-tested-macos.txt", "LICENSE"]:
        shutil.copy2(root / name, snapshot / name)
    private = output / "private"
    private.mkdir()
    library = private / "shapes.h5"
    shutil.copy2(source_library, library)
    if sha256(library) != study["library_sha256"]:
        raise ValueError("Input changed while freezing")
    shutil.copy2(args.protocol, snapshot / "protocol-original.json")
    study["library"] = str(library)
    resolved = snapshot / "protocol-resolved.json"
    write_json(resolved, study)
    manifest = {
        "format": "sand-dem-frozen-performance/1",
        "original_protocol_sha256": sha256(args.protocol),
        "sources": {
            str(f.relative_to(snapshot)): sha256(f)
            for f in snapshot.rglob("*")
            if f.is_file()
        },
        "shape_sha256": sha256(library),
        "status": "running",
    }
    write_json(output / "frozen-manifest.json", manifest)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(snapshot)
    start = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            str(snapshot / "frozen_benchmark.py"),
            str(resolved),
            "--output",
            str(output),
            "--execute-frozen",
        ],
        env=env,
        check=False,
    )
    manifest.update(
        status="complete" if completed.returncode == 0 else "failed",
        return_code=completed.returncode,
        elapsed_wall_seconds=time.perf_counter() - start,
    )
    write_json(output / "frozen-manifest.json", manifest)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
