"""Frozen current-solver runtime profile of a saved checkpoint, serial Metal only.

Synchronized stage measurements are wall times, never labelled GPU timestamps.
"""

import argparse
import cProfile
import json
import os
import pstats
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from sand_atlas_dem.util import sha256, write_json


def profile_rows(profiler):
    stats = pstats.Stats(profiler)
    rows = []
    for (file, line, function), (
        primitive,
        calls,
        own,
        cumulative,
        _,
    ) in stats.stats.items():
        rows.append(
            dict(
                file=file,
                line=line,
                function=function,
                calls=calls,
                primitive_calls=primitive,
                own_seconds=own,
                cumulative_seconds=cumulative,
            )
        )
    return sorted(rows, key=lambda r: r["cumulative_seconds"], reverse=True)[:60]


def summary(values):
    return dict(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        count=len(values),
    )


def split_step(e, policy):
    """Same kernel order as Engine.profile_step; sync at subsystem boundaries."""
    b = e.bindings(drag=policy["drag_s"])
    stages = []

    def stage(name, encode):
        start = time.perf_counter()
        encoder = e.m.device.create_command_encoder()
        encode(encoder)
        encoded = time.perf_counter()
        e.m.submit(encoder)
        e.m.device.wait()
        end = time.perf_counter()
        stages.append(
            dict(
                name=name,
                encode_seconds=encoded - start,
                submit_wait_seconds=end - encoded,
                wall_seconds=end - start,
            )
        )

    def dispatch(names, encoder):
        for name, n in names:
            e.dispatch(name, n, encoder, b)

    def neighbours(encoder):
        dispatch(
            [("clearBins", e.bin_count), ("fillBins", e.n), ("countPairs", e.n)],
            encoder,
        )
        e.scan_pairs.encode(encoder, e.buffers["pairCounts"], e.buffers["pairOffsets"])
        dispatch([("emitPairs", e.n)], encoder)

    stage(
        "drift",
        lambda enc: dispatch(
            [("beginStep", e.n), ("advanceCell", 1), ("checkRebuild", e.n)], enc
        ),
    )
    stage("neighbours", neighbours)
    stage("contact_geometry", lambda enc: e._encode_contacts(enc, b))
    stage(
        "friction_and_history",
        lambda enc: dispatch(
            [
                ("clearContactStorage", max(e.n, e.hash_size)),
                ("resolveContacts", e.contact_capacity),
                ("buildHistoryHash", e.contact_capacity),
            ],
            enc,
        ),
    )
    stage("force_reduction", lambda enc: dispatch([("gather", e.n)], enc))
    stage("final_kick", lambda enc: dispatch([("finishStep", e.n)], enc))
    e.check_errors()
    e.active = 1 - e.active
    e.history = True
    e.step += 1
    return stages


def execute(out):
    policy = json.loads((out / "protocol.json").read_text())
    freeze = json.loads((out / "prospective-freeze.json").read_text())
    snapshot = out / "source_snapshot"
    assert sha256(out / "protocol.json") == freeze["protocol_sha256"]
    assert all(sha256(snapshot / p) == h for p, h in freeze["sources"].items())
    assert all(sha256(p) == h for p, h in freeze["inputs"].items())
    from sand_atlas_dem.benchmark import memory_report
    from sand_atlas_dem.simulation import Simulation

    startup = cProfile.Profile()
    start = time.perf_counter()
    sim = startup.runcall(Simulation.resume, policy["checkpoint"])
    e = sim.engine
    startup_seconds = time.perf_counter() - start
    write_json(
        out / "startup-profile.json",
        dict(wall_seconds=startup_seconds, functions=profile_rows(startup)),
    )
    assert (
        e.spec.timestep_policy == "conservative" and not e.wall_count and not e.viscous
    )
    with h5py.File(policy["checkpoint"]) as f:
        state = {name: f[name][:] for name in f}
        state["contacts"] = int(f.attrs["contacts"])
        step = int(f.attrs["step"])

    def reset():
        e.restore(state, step)
        e.advance(policy["warmup_steps"], rate=policy["rate_s"], drag=policy["drag_s"])

    batches = []
    for repeat in range(policy["repetitions"]):
        for count in policy["batch_steps"]:
            reset()
            start = time.perf_counter()
            e.advance(count, rate=policy["rate_s"], drag=policy["drag_s"])
            elapsed = time.perf_counter() - start
            batches.append(
                dict(
                    repeat=repeat,
                    steps=count,
                    wall_seconds=elapsed,
                    seconds_per_step=elapsed / count,
                    diagnostics=e.diagnostics(),
                )
            )
            write_json(out / "batches.json", batches)
    reset()
    host = cProfile.Profile()
    host.runcall(
        e.advance,
        policy["host_profile_steps"],
        rate=policy["rate_s"],
        drag=policy["drag_s"],
    )
    write_json(out / "host-profile.json", dict(functions=profile_rows(host)))

    # Same checkpoint + warmup, ordinary batched advance for perturbation check.
    reset()
    e.advance(policy["split_steps"], rate=policy["rate_s"], drag=policy["drag_s"])
    reference = e.state()
    splits = []
    differences = []
    for repeat in range(policy["repetitions"]):
        reset()
        for index in range(policy["split_steps"]):
            splits.append(
                dict(repeat=repeat, index=index, stages=split_step(e, policy))
            )
        observed = e.state()
        errors = {}
        for name in ["position", "velocity", "force", "rotation"]:
            a = observed[name].astype(float)
            b = reference[name].astype(float)
            residual = name + "Remainder"
            if residual in observed:
                a += observed[residual]
                b += reference[residual]
            errors[name] = float(np.max(abs(a - b)))
        differences.append(errors)
        write_json(out / "split-stages.json", splits)
    summaries = {
        str(count): summary(
            [x["seconds_per_step"] for x in batches if x["steps"] == count]
        )
        for count in policy["batch_steps"]
    }
    names = [row["name"] for row in splits[0]["stages"]]
    stage_summary = {
        name: {
            field: summary(
                [
                    r[field]
                    for sample in splits
                    for r in sample["stages"]
                    if r["name"] == name
                ]
            )
            for field in ["wall_seconds", "encode_seconds", "submit_wait_seconds"]
        }
        for name in names
    }
    unchanged = all(
        sha256(snapshot / p) == h for p, h in freeze["sources"].items()
    ) and all(sha256(p) == h for p, h in freeze["inputs"].items())
    equivalent = all(
        error <= policy["partition_state_max_absolute_error"]
        for result in differences
        for error in result.values()
    )
    write_json(
        out / "report.json",
        dict(
            identity=policy["identity"],
            completed=True,
            backend=e.m.doctor(),
            sources_and_inputs_unchanged=unchanged,
            startup_seconds=startup_seconds,
            batch_seconds_per_step=summaries,
            stage_synchronized_wall=stage_summary,
            partition_state_max_absolute_errors=differences,
            partition_check_passed=equivalent,
            memory=memory_report(e),
            scientific_admission=False,
            timing_scope="Wall-clock encoding and submitted device execution with explicit stage synchronization. Not GPU-only timings; do not add stage costs to predict unpartitioned runtime.",
        ),
    )
    return 0 if unchanged and equivalent else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-frozen", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.execute_frozen:
        return execute(out)
    root = Path(__file__).resolve().parents[1]
    policy = json.loads(args.protocol.read_text())
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        *root.glob("src/**/*.py"),
        *root.glob("src/**/*.slang"),
        Path(__file__).resolve(),
        root / "pyproject.toml",
    ]
    sources = {}
    for p in sorted(paths):
        relative = (
            Path("profile_runtime.py")
            if p == Path(__file__).resolve()
            else p.relative_to(root / "src")
            if "src" in p.relative_to(root).parts[:1]
            else Path(p.name)
        )
        dest = out / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        sources[str(relative)] = sha256(dest)
    shutil.copyfile(args.protocol, out / "protocol.json")
    inputs = {p: sha256(p) for p in [policy["checkpoint"], policy["shape_library"]]}
    assert inputs[policy["checkpoint"]] == policy["checkpoint_sha256"]
    assert inputs[policy["shape_library"]] == policy["shape_sha256"]
    write_json(
        out / "prospective-freeze.json",
        dict(
            protocol_sha256=sha256(out / "protocol.json"),
            sources=sources,
            inputs=inputs,
        ),
    )
    with (out / "run.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(out / "source_snapshot/profile_runtime.py"),
                str(out / "protocol.json"),
                "--output",
                str(out),
                "--execute-frozen",
            ],
            env=dict(os.environ, PYTHONPATH=str(out / "source_snapshot")),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
