"""Archive executable sources and input bytes, then run an immutable controller study."""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunks-per-checkpoint", type=int, default=100)
    p.add_argument("--resume", type=Path)
    p.add_argument("--source-snapshot", type=Path)
    p.add_argument("--extend-step-budgets", action="store_true")
    args = p.parse_args()
    if args.chunks_per_checkpoint < 1:
        raise ValueError("Positive checkpoint interval required")
    root = Path(__file__).resolve().parents[1]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    source_root = root
    package = root / "src/sand_atlas_dem"
    if args.source_snapshot:
        source_root = args.source_snapshot.resolve()
        parent_manifest = source_root.parent / "frozen-manifest.json"
        receipt = read_json(parent_manifest)
        for name, digest in receipt["sources"].items():
            if sha256(source_root / name) != digest:
                raise ValueError(
                    "Requested source snapshot differs from its frozen receipt"
                )
        sha256(parent_manifest)
        package = source_root / "sand_atlas_dem"
    shutil.copytree(
        package,
        snapshot / "sand_atlas_dem",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(
        root / "examples/controller_study.py", snapshot / "controller_study.py"
    )
    for name in ["pyproject.toml", "requirements-tested-macos.txt", "LICENSE"]:
        shutil.copy2(source_root / name, snapshot / name)
    spec = Experiment.load(args.experiment)
    original_shape = Path(spec.shape_library)
    private = args.output / "private"
    private.mkdir()
    shape = private / "shapes.h5"
    shutil.copy2(original_shape, shape)
    if sha256(shape) != sha256(original_shape):
        raise ValueError("Input changed while freezing")
    spec.shape_library = str(shape)
    protocol = snapshot / "experiment-resolved.json"
    # Preserve the source protocol fields. Defaults belong to the frozen solver
    # version, not to the wrapper's potentially newer installed dataclass.
    resolved_spec = read_json(args.experiment)
    resolved_spec["shape_library"] = str(shape)
    write_json(protocol, resolved_spec)
    shutil.copy2(args.experiment, snapshot / "experiment-original.json")
    manifest = {
        "format": "sand-dem-frozen-study/1",
        "original_experiment_path": str(args.experiment.resolve()),
        "original_experiment_sha256": sha256(args.experiment),
        "original_shape_path": str(original_shape),
        "shape_sha256": sha256(shape),
        "sources": {
            str(f.relative_to(snapshot)): sha256(f)
            for f in snapshot.rglob("*")
            if f.is_file()
        },
        "environment": "Repository Python environment; exact tested dependency list archived",
        "status": "running",
    }
    write_json(args.output / "frozen-manifest.json", manifest)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(snapshot)
    cmd = [
        sys.executable,
        str(snapshot / "controller_study.py"),
        str(protocol),
        "--output",
        str(args.output / "run"),
        "--chunks-per-checkpoint",
        str(args.chunks_per_checkpoint),
    ]
    if args.resume:
        parent = private / "parent-checkpoint.h5"
        shutil.copy2(args.resume, parent)
        if sha256(parent) != sha256(args.resume):
            raise ValueError("Parent checkpoint changed while freezing")
        manifest["parent_checkpoint_sha256"] = sha256(parent)
        cmd.extend(["--resume", str(parent)])
    if args.extend_step_budgets:
        if args.resume is None:
            raise ValueError("A budget extension requires a parent checkpoint")
        cmd.append("--extend-step-budgets")
    write_json(args.output / "frozen-manifest.json", manifest)
    start = time.perf_counter()
    completed = subprocess.run(cmd, env=env, check=False)
    manifest.update(
        status="complete" if completed.returncode == 0 else "failed",
        return_code=completed.returncode,
        elapsed_wall_seconds=time.perf_counter() - start,
    )
    write_json(args.output / "frozen-manifest.json", manifest)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
