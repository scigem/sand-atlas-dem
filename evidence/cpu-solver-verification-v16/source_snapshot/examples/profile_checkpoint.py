"""Freeze a completed checkpoint's original solver for explicit serial profiling."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def execute(protocol_path, output):
    # Only the frozen subprocess imports a solver; the resolver never starts Metal.
    import time

    freeze = read(protocol_path.parent / "prospective-freeze.json")
    if digest(protocol_path) != freeze["protocol_sha256"]:
        raise ValueError("Frozen profiling protocol changed")
    protocol = read(protocol_path)
    if protocol["source_sha256"] != freeze["source_sha256"]:
        raise ValueError("Frozen profiling source receipt changed")
    inputs = protocol["input_sha256"]
    if any(digest(p) != value for p, value in inputs.items()):
        raise ValueError("Frozen profiling input changed")
    snapshot = protocol_path.parent / "source_snapshot"
    if any(
        digest(snapshot / n) != value for n, value in protocol["source_sha256"].items()
    ):
        raise ValueError("Frozen profiling source changed")
    import numpy as np

    from sand_atlas_dem.benchmark import memory_report
    from sand_atlas_dem.simulation import Simulation

    output.mkdir(parents=True, exist_ok=False)
    sim = Simulation.resume(protocol["checkpoint"])
    engine = sim.engine
    backend = engine.m.doctor()
    with h5py.File(protocol["checkpoint"]) as f:
        state = {name: f[name][:] for name in f}
        state["contacts"] = int(f.attrs["contacts"])
        step = int(f.attrs["step"])
    trials = []
    for repeat in range(protocol["repetitions"]):
        engine.restore(state, step)
        engine.advance(
            protocol["warmup_steps"], rate=protocol["rate_s"], drag=protocol["drag_s"]
        )
        previous_bindings = engine.bindings

        # Original archived profile_step uses bindings() without drag. Preserve
        # its exact dispatch sequence while supplying the declared physical drag.
        def bindings(dt=None, drag=protocol["drag_s"]):
            return previous_bindings(dt=dt, drag=drag)

        engine.bindings = bindings
        samples = []
        try:
            for _ in range(protocol["profile_steps"]):
                start = time.perf_counter()
                timing = engine.profile_step()
                samples.append(
                    dict(wall_seconds=time.perf_counter() - start, gpu=timing)
                )
        finally:
            del engine.bindings
        trial = dict(
            repetition=repeat,
            initial_step=step,
            final_step=engine.step,
            diagnostics=engine.diagnostics(),
            samples=samples,
        )
        trials.append(trial)
        write(output / f"trial-{repeat:02d}.json", trial)
    samples = [s for t in trials for s in t["samples"]]
    available = [s["gpu"] for s in samples if "seconds" in s["gpu"]]
    gpu_means = (
        {
            name: float(np.mean([s["seconds"][name] for s in available]))
            for name in available[0]["seconds"]
        }
        if available
        else {}
    )
    unchanged = all(digest(p) == value for p, value in inputs.items()) and all(
        digest(snapshot / n) == value for n, value in protocol["source_sha256"].items()
    )
    write(
        output / "report.json",
        dict(
            identity=protocol["identity"],
            completed=True,
            inputs_unchanged=unchanged,
            preparation_accepted=False,
            scientific_admission=False,
            gpu_executed=True,
            backend=backend,
            profiled_samples=len(samples),
            gpu_timestamp_samples=len(available),
            mean_step_wall_seconds=float(np.mean([s["wall_seconds"] for s in samples])),
            mean_gpu_subsystem_seconds=gpu_means,
            memory_accounting=memory_report(engine),
            scope=protocol["scope"],
        ),
    )
    return 0 if unchanged else 1


def resolve(policy_path, output):
    policy_digest = digest(policy_path)
    driver_digest = digest(Path(__file__))
    policy = read(policy_path)
    for name in ("warmup_steps", "profile_steps", "repetitions"):
        if type(policy[name]) is not int or policy[name] < 1:
            raise ValueError("Positive integer profiling workloads required")
    checkpoint = Path(policy["checkpoint"]).resolve()
    if digest(checkpoint) != policy["checkpoint_sha256"]:
        raise ValueError("Completed checkpoint changed")
    parent = Path(policy["preparation_study"]).resolve()
    receipt = read(parent / "frozen-manifest.json")
    sources = {}
    for name, value in receipt["sources"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe archived source path")
        source = parent / "source_snapshot" / relative
        if digest(source) != value:
            raise ValueError("Preparation source archive changed")
        sources[name] = source
    required = {
        "sand_atlas_dem/" + name
        for name in (
            "__init__.py",
            "gpu.py",
            "backend.py",
            "simulation.py",
            "geometry.py",
            "spec.py",
            "benchmark.py",
            "shaders/solver.slang",
        )
    }
    if not required.issubset(sources):
        raise ValueError("Complete original profiling solver archive required")
    with h5py.File(checkpoint) as f:
        spec = json.loads(f.attrs["experiment"])
        if int(f.attrs["step"]) != policy["checkpoint_step"] or not bool(
            f.attrs["contact_phase"]
        ):
            raise ValueError("Expected completed contact checkpoint required")
        if int(f.attrs["stage_index"]) != 0:
            raise ValueError("Original isotropic preparation stage required")
        shape = Path(spec["shape_library"]).resolve()
        shape_digest = digest(shape)
        if (
            shape_digest != f.attrs["shape_sha256"]
            or shape_digest != receipt["shape_sha256"]
        ):
            raise ValueError("Checkpoint shape asset changed")
    if (
        spec["contact_geometry"] != "mesh"
        or spec["particles"] != policy["particles"]
        or not spec["periodic"]
        or spec["planes"]
        or spec.get("normal_viscosity", 0) != 0
        or spec["stages"][0]["kind"] != "isotropic"
        or spec["stages"][0]["drag"] != policy["drag_s"]
    ):
        raise ValueError("Declared undamped-contact periodic mesh workload required")
    import math

    rate = policy["rate_s"]
    if len(rate) != 3 or any(not math.isfinite(x) for x in rate):
        raise ValueError("Three finite prescribed rates required")
    cap = min(spec["stages"][0]["max_rate"], spec["stages"][0]["contact_max_rate"])
    if any(abs(x) > cap for x in rate):
        raise ValueError("Profiling rate exceeds frozen contact cap")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    for name, source in sources.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != receipt["sources"][name]:
            raise ValueError("Preparation source changed while freezing")
    driver = snapshot / "profile_checkpoint.py"
    if driver.exists():
        raise ValueError("Source archive collides with profiling driver")
    shutil.copy2(Path(__file__), driver)
    if digest(driver) != driver_digest:
        raise ValueError("Profiling driver changed while freezing")
    (output / "private").mkdir(mode=0o700)
    frozen_checkpoint = output / "private/checkpoint.h5"
    shutil.copy2(checkpoint, frozen_checkpoint)
    if digest(frozen_checkpoint) != policy["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed while freezing")
    shutil.copy2(
        parent / "frozen-manifest.json", output / "parent-manifest-observed.json"
    )
    shutil.copy2(policy_path, output / "policy.json")
    if (
        digest(output / "policy.json") != policy_digest
        or digest(policy_path) != policy_digest
    ):
        raise ValueError("Profiling policy changed while freezing")
    source_hashes = {
        str(p.relative_to(snapshot)): digest(p)
        for p in snapshot.rglob("*")
        if p.is_file()
    }
    protocol = {
        **policy,
        "checkpoint": str(frozen_checkpoint),
        "shape_library": str(shape),
        "input_sha256": {
            str(frozen_checkpoint): digest(frozen_checkpoint),
            str(shape): shape_digest,
        },
        "source_sha256": source_hashes,
        "scientific_admission": False,
        "preparation_accepted": False,
    }
    write(output / "protocol.json", protocol)
    write(
        output / "prospective-freeze.json",
        dict(
            policy_sha256=policy_digest,
            protocol_sha256=digest(output / "protocol.json"),
            source_sha256=source_hashes,
            gpu_executed=False,
            scope=policy["scope"],
        ),
    )
    return driver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--execute-frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.execute_frozen:
        return execute(args.policy.resolve(), output)
    driver = resolve(args.policy.resolve(), output)
    if not args.run:
        return 0
    with (output / "profile.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(driver),
                str(output / "protocol.json"),
                "--execute-frozen",
                "--output",
                str(output / "profile"),
            ],
            env={**os.environ, "PYTHONPATH": str(output / "source_snapshot")},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
