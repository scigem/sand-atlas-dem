"""Frozen adaptive triangle-tree experiment, serial actual Metal only."""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import h5py
import numpy as np

from sand_atlas_dem.util import sha256, write_json


def execute(out):
    from benchmark_parallel_mesh import compare
    from immutable_bindings import BoundKernel, ImmutableBindings
    from profile_runtime import split_step

    from sand_atlas_dem.simulation import Simulation

    p = json.loads((out / "protocol.json").read_text())
    f = json.loads((out / "prospective-freeze.json").read_text())
    snapshot = out / "source_snapshot"

    def intact():
        return (
            sha256(out / "protocol.json") == f["protocol_sha256"]
            and all(sha256(snapshot / n) == h for n, h in f["sources"].items())
            and all(sha256(n) == h for n, h in f["inputs"].items())
        )

    assert intact()
    e = Simulation.resume(p["checkpoint"]).engine
    assert (
        not e.wall_count and not e.viscous and e.spec.timestep_policy == "conservative"
    )
    with h5py.File(p["checkpoint"]) as h:
        saved = {name: h[name][:] for name in h}
        saved["contacts"] = int(h.attrs["contacts"])
        step = int(h.attrs["step"])
    from adaptive_mesh import build_bricks

    roots = e.m.read(e.buffers["meshRoots"], np.uint32).reshape(-1, 4)
    links = e.m.read(e.buffers["meshLinks"], np.uint32).reshape(-1, 4)
    triangles = e.m.read(e.buffers["meshTriangles"]).reshape(-1, 3, 4)
    arrays, receipt = build_bricks(
        roots,
        links,
        triangles,
        resolution=p["brick_resolution"],
        capacity=p["brick_capacity"],
    )
    assert e.estimated_bytes + receipt["bytes"] < e.spec.working_memory_gib * 1024**3
    write_json(out / "brick-build.json", receipt)
    brick_buffers = {name: e.m.buffer(a) for name, a in arrays.items()}
    binding_cache = ImmutableBindings(e.m)
    native_kernel = e.m.kernel
    wrappers = {}

    def kernel(self, name, module="solver.slang"):
        key = (name, str(module))
        if key not in wrappers:
            wrappers[key] = BoundKernel(binding_cache, native_kernel(name, module))
        return wrappers[key]

    e.m.kernel = types.MethodType(kernel, e.m)
    ordinary_dispatch = e.dispatch
    mode = "baseline"
    metrics = e.m.buffer(np.zeros(9, np.uint32))

    def dispatch(self, name, n, encoder, bindings):
        if name in {"prepareMeshPairs", "collectMeshFrontiers"}:
            values = (
                dict(bindings, meshQueryMetrics=metrics)
                if mode.endswith("-metrics")
                else bindings
            )
            if mode.startswith("brick"):
                values = dict(values, **brick_buffers)
            module = snapshot / "sand_atlas_dem/shaders" / (mode + ".slang")
            self.m.kernel(name, str(module)).dispatch(
                thread_count=[max(1, int(n)), 1, 1],
                command_encoder=encoder,
                vars={"r": values},
            )
        else:
            ordinary_dispatch(name, n, encoder, bindings)

    e.dispatch = types.MethodType(dispatch, e)

    def reset():
        e.restore(saved, step)
        e.advance(p["warmup_steps"], rate=p["rate_s"], drag=p["drag_s"])

    records, checks, states = [], [], {}
    for repeat in range(p["repetitions"]):
        for order in [p["modes"], list(reversed(p["modes"]))]:
            for mode in order:
                reset()
                start = time.perf_counter()
                e.advance(p["steps"], rate=p["rate_s"], drag=p["drag_s"])
                elapsed = (time.perf_counter() - start) / p["steps"]
                current = e.state()
                if mode in states:
                    checks.append(
                        dict(
                            mode=mode,
                            kind="repeat",
                            **compare(states[mode], current, p),
                        )
                    )
                else:
                    states[mode] = current
                records.append(
                    {"mode": mode, "repeat": repeat, "seconds_per_step": elapsed}
                )
                write_json(out / "samples.json", records)
    checks.append(
        dict(
            mode="brick",
            kind="baseline-comparison",
            **compare(states["baseline"], states["brick"], p),
        )
    )
    stages, counters = {}, {}
    for base in p["modes"]:
        mode = base
        reset()
        initial, initial_step = e.state(), e.step
        e.advance(p["split_steps"], rate=p["rate_s"], drag=p["drag_s"])
        expected = e.state()
        e.restore(initial, initial_step)
        # split_step assumes advanceCell applies the currently configured rate.
        stages[base] = [split_step(e, p) for _ in range(p["split_steps"])]
        checks.append(
            dict(mode=base, kind="split-comparison", **compare(expected, e.state(), p))
        )
        mode = base + "-metrics"
        reset()
        metrics.copy_from_numpy(np.zeros(9, np.uint32))
        e.advance(1, rate=p["rate_s"], drag=p["drag_s"])
        counters[base] = dict(
            zip(
                ["queries", "bvh_nodes", "triangles", "seeded_queries"],
                map(int, e.m.read(metrics, np.uint32)[:4]),
            )
        )
        extra = e.m.read(metrics, np.uint32)
        counters[base].update(
            list_queries=int(extra[4]),
            fast_queries=int(extra[5]),
            list_triangles=int(extra[6]),
            tie_fallbacks=int(extra[8]),
        )
    write_json(out / "split-stages.json", stages)
    medians = {
        m: statistics.median(r["seconds_per_step"] for r in records if r["mode"] == m)
        for m in p["modes"]
    }
    speedup = medians["baseline"] / medians["brick"]
    passed = all(c["passed"] for c in checks)
    write_json(
        out / "report.json",
        {
            "identity": p["identity"],
            "completed": True,
            "backend": e.m.doctor(),
            "sources_and_inputs_unchanged": intact(),
            "median_seconds_per_step": medians,
            "speedup": speedup,
            "meets_2x": passed and speedup >= 2,
            "eligible": passed and speedup >= p["minimum_speedup"],
            "equivalence_passed": passed,
            "checks": checks,
            "counters": counters,
            "brick_build": receipt,
            "stage_median_seconds": {
                m: {
                    name: statistics.median(
                        s["wall_seconds"]
                        for row in rows
                        for s in row
                        if s["name"] == name
                    )
                    for name in [s["name"] for s in rows[0]]
                }
                for m, rows in stages.items()
            },
            "stage_scope": "Synchronized wall times, not GPU timestamps; immutable binding reuse enabled equally in both modes.",
            "scientific_admission": False,
        },
    )
    assert intact() and passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.execute:
        execute(out)
        return
    from adaptive_mesh import brick_shader
    from coherent_mesh_shader import instrument

    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    p = json.loads(args.protocol.read_text())
    write_json(out / "protocol.json", p)
    paths = (
        list((root / "src").rglob("*.py"))
        + list((root / "src").rglob("*.slang"))
        + [Path(__file__).resolve()]
        + [
            root / "examples" / n
            for n in [
                "adaptive_mesh.py",
                "brick_mesh.py",
                "coherent_mesh_shader.py",
                "benchmark_parallel_mesh.py",
                "profile_runtime.py",
                "immutable_bindings.py",
            ]
        ]
    )
    hashes = {}
    for path in paths:
        relative = (
            path.relative_to(root / "src")
            if path.is_relative_to(root / "src")
            else Path(path.name)
        )
        dest = out / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        hashes[str(relative)] = sha256(dest)
    source = (root / "src/sand_atlas_dem/shaders/solver.slang").read_text()
    for name, text in {"baseline": source, "brick": brick_shader(source)}.items():
        for suffix, shader in [
            ("", text),
            (
                "-metrics",
                brick_shader(source, metrics=True)
                if name == "brick"
                else instrument(text),
            ),
        ]:
            relative = Path("sand_atlas_dem/shaders") / (name + suffix + ".slang")
            dest = out / "source_snapshot" / relative
            dest.write_text(shader)
            hashes[str(relative)] = sha256(dest)
    inputs = {p[k]: sha256(p[k]) for k in ["checkpoint", "shape_library"]}
    assert inputs[p["checkpoint"]] == p["checkpoint_sha256"]
    assert inputs[p["shape_library"]] == p["shape_sha256"]
    write_json(
        out / "prospective-freeze.json",
        {
            "sources": hashes,
            "inputs": inputs,
            "protocol_sha256": sha256(out / "protocol.json"),
        },
    )
    with (out / "run.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(out / "source_snapshot" / Path(__file__).name),
                str(out / "protocol.json"),
                "--output",
                str(out),
                "--execute",
            ],
            env=dict(os.environ, PYTHONPATH=str(out / "source_snapshot")),
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
