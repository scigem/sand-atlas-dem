"""Production fine-frontier collector vs archived32-frontier kernel, frozen A/B."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import h5py
import numpy as np

from sand_atlas_dem.util import sha256, write_json


def baseline_contacts(e, encoder, bindings):
    e.dispatch("resetMeshContacts", e.pair_capacity, encoder, bindings)
    e.dispatch("prepareMeshPairs", e.pair_capacity, encoder, bindings)
    reference = Path(__file__).with_name("reference-32.slang")
    e.m.kernel("collectMeshFrontiers", str(reference)).dispatch(
        thread_count=[64 * 4096, 1, 1], command_encoder=encoder, vars={"r": bindings}
    )


def compare(a, b, policy):
    errors = {}
    for name in ["position", "velocity", "rotation"]:
        x = a[name].astype(float)
        y = b[name].astype(float)
        if name + "Remainder" in a:
            x += a[name + "Remainder"]
            y += b[name + "Remainder"]
        errors[name] = float(np.max(abs(x - y)))
    for name in ["force", "torque"]:
        scale = max(float(np.max(np.linalg.norm(a[name][:, :3], axis=1))), 1e-12)
        errors[name] = float(
            np.max(np.linalg.norm(a[name][:, :3] - b[name][:, :3], axis=1)) / scale
        )
    ai = np.lexsort(a["meta"][:, :3].T[::-1])
    bi = np.lexsort(b["meta"][:, :3].T[::-1])
    keys = np.array_equal(a["meta"][ai], b["meta"][bi])
    history = keys and np.allclose(
        a["slip"][ai],
        b["slip"][bi],
        rtol=policy["history_rtol"],
        atol=policy["history_atol"],
    )
    normals = keys and np.allclose(
        a["normalArea"][ai], b["normalArea"][bi], rtol=1e-6, atol=1e-7
    )
    passed = (
        keys
        and history
        and normals
        and all(v <= policy["limits"][k] for k, v in errors.items())
    )
    return dict(
        passed=bool(passed),
        errors=errors,
        keys_equal=keys,
        history_equal=bool(history),
        normals_equal=bool(normals),
    )


def execute(out):
    from sand_atlas_dem.simulation import Simulation

    p = json.loads((out / "protocol.json").read_text())
    freeze = json.loads((out / "prospective-freeze.json").read_text())
    assert sha256(out / "protocol.json") == freeze["protocol_sha256"]
    assert all(
        sha256(out / "source_snapshot" / n) == h for n, h in freeze["sources"].items()
    )
    assert all(sha256(n) == h for n, h in freeze["inputs"].items())
    start = time.perf_counter()
    sim = Simulation.resume(p["checkpoint"])
    engine = sim.engine
    setup = time.perf_counter() - start
    assert (
        not engine.wall_count
        and not engine.viscous
        and engine.spec.timestep_policy == "conservative"
    )
    with h5py.File(p["checkpoint"]) as f:
        state = {name: f[name][:] for name in f}
        state["contacts"] = int(f.attrs["contacts"])
        step = int(f.attrs["step"])
    assert engine.mesh_frontier_count == 128
    fine_frontier = engine.buffers["meshFrontier"]
    ranges = engine.m.read(engine.buffers["meshClusterRanges"], np.uint32).reshape(
        -1, 4
    )
    info = engine.m.read(engine.buffers["meshClusterInfo"], np.uint32).reshape(-1, 4)
    entries = []
    for first, *_ in info:
        pending = [int(first)]
        while len(pending) < 32:
            splittable = [n for n in pending if ranges[n, 1] == 0]
            if not splittable:
                break
            node = max(splittable, key=lambda n: int(ranges[n, 2]) - n)
            left = node + 1
            at = pending.index(node)
            pending[at : at + 1] = [left, int(ranges[left, 2])]
        entries.extend([[n, int(ranges[n, 2]), 0, 0] for n in pending])
        entries.extend([[0, 0, 0, 0]] * (32 - len(pending)))
    baseline_frontier = engine.m.buffer(np.asarray(entries, np.uint32))
    parallel = engine._encode_contacts
    samples = []
    states = {}
    checks = []
    for repetition in range(p["repetitions"]):
        for mode in p["mode_order"]:
            engine._encode_contacts = (
                parallel
                if mode == "parallel"
                else types.MethodType(baseline_contacts, engine)
            )
            engine.buffers["meshFrontier"] = (
                fine_frontier if mode == "parallel" else baseline_frontier
            )
            engine.restore(state, step)
            engine.advance(p["warmup_steps"], rate=p["rate_s"], drag=p["drag_s"])
            start = time.perf_counter()
            engine.advance(p["steps"], rate=p["rate_s"], drag=p["drag_s"])
            elapsed = time.perf_counter() - start
            current = engine.state()
            if mode not in states:
                states[mode] = current
            else:
                checks.append(compare(states[mode], current, p))
            samples.append(
                dict(
                    repetition=repetition,
                    mode=mode,
                    wall_seconds=elapsed,
                    seconds_per_step=elapsed / p["steps"],
                    diagnostics=engine.diagnostics(),
                )
            )
            write_json(out / "samples.json", samples)
    checks.append(compare(states["baseline"], states["parallel"], p))
    medians = {
        mode: float(
            np.median([s["seconds_per_step"] for s in samples if s["mode"] == mode])
        )
        for mode in states
    }
    speedup = medians["baseline"] / medians["parallel"]
    unchanged = all(
        sha256(out / "source_snapshot" / n) == h for n, h in freeze["sources"].items()
    ) and all(sha256(n) == h for n, h in freeze["inputs"].items())
    passed = unchanged and all(c["passed"] for c in checks)
    write_json(
        out / "report.json",
        dict(
            identity=p["identity"],
            completed=True,
            backend=engine.m.doctor(),
            setup_seconds=setup,
            sources_and_inputs_unchanged=unchanged,
            equivalence_checks=checks,
            equivalence_passed=passed,
            median_seconds_per_step=medians,
            speedup=speedup,
            performance_passed=speedup >= p["minimum_speedup"],
            scientific_admission=False,
            scope=p["scope"],
        ),
    )
    return 0 if passed and speedup >= p["minimum_speedup"] else 1


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
    out.mkdir(parents=True, exist_ok=False)
    p = json.loads(args.protocol.read_text())
    sources = {}
    for file in sorted(
        [
            *root.glob("src/**/*.py"),
            *root.glob("src/**/*.slang"),
            Path(__file__).resolve(),
        ]
    ):
        relative = (
            Path("benchmark_promoted_mesh.py")
            if file == Path(__file__).resolve()
            else file.relative_to(root / "src")
        )
        dest = out / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest)
        sources[str(relative)] = sha256(dest)
    reference = Path(p["reference_shader"])
    assert sha256(reference) == p["reference_shader_sha256"]
    dest = out / "source_snapshot/reference-32.slang"
    shutil.copyfile(reference, dest)
    sources["reference-32.slang"] = sha256(dest)
    shutil.copyfile(args.protocol, out / "protocol.json")
    inputs = {file: sha256(file) for file in [p["checkpoint"], p["shape_library"]]}
    assert (
        inputs[p["checkpoint"]] == p["checkpoint_sha256"]
        and inputs[p["shape_library"]] == p["shape_sha256"]
    )
    write_json(
        out / "prospective-freeze.json",
        dict(
            sources=sources,
            inputs=inputs,
            protocol_sha256=sha256(out / "protocol.json"),
        ),
    )
    with (out / "run.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(out / "source_snapshot/benchmark_promoted_mesh.py"),
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
