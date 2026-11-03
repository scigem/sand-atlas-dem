"""Frozen scheduling/rejection A/B trials plus untimed traversal counters, Metal only."""

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


def variants(source):
    from segmented_shader import precision, segmented

    corrected = precision(source)
    return {
        "baseline": (source, 4096),
        "precision": (corrected, 4096),
        "segmented": (segmented(corrected), 4096),
        "segmented-original": (segmented(source), 4096),
    }


def execute(out):
    from benchmark_parallel_mesh import compare

    from sand_atlas_dem.backend import Metal
    from sand_atlas_dem.geometry import ShapeLibrary, from_mesh, with_face_quadrature
    from sand_atlas_dem.gpu import Engine, f4
    from sand_atlas_dem.simulation import Simulation
    from sand_atlas_dem.spec import Experiment

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
    if p["workload"] == "hostun":
        e = Simulation.resume(p["checkpoint"]).engine
        with h5py.File(p["checkpoint"]) as h:
            saved = {name: h[name][:] for name in h}
            saved["contacts"] = int(h.attrs["contacts"])
            step = int(h.attrs["step"])
    else:
        import trimesh

        shape = with_face_quadrature(
            from_mesh(
                trimesh.creation.box(extents=[100] * 3),
                unit="um",
                resolution=8,
                samples=32,
            ),
            rule="gauss3",
            subdivisions=p["subdivisions"],
        )
        side = p.get("grid_side", 3)
        positions = 0.005 + np.indices((side, side, side)).reshape(3, -1).T * 0.999e-4
        count = len(positions)
        e = Engine(
            ShapeLibrary([shape]),
            Experiment(
                periodic=False,
                contact_geometry="mesh",
                particles=count,
                timestep=1e-9,
                contacts_per_particle=8192,
                pairs_per_particle=32,
                bin_capacity=64,
                max_overlap_fraction=0.1,
            ),
            positions,
            [0.01] * 3,
            [0] * count,
            [1] * count,
            metal=Metal(),
        )
        if "omega_rad_s" in p:
            inertia = shape.inertia * e.spec.density
            momentum = (
                inertia
                @ np.asarray(p["omega_rad_s"])
                / (e.mass_unit * e.length_unit**2 / e.time_unit)
            )
            e.buffers["momentum"].copy_from_numpy(f4(np.tile(momentum, (e.n, 1))))
        e.initialize()
        saved, step = e.state(), 0
    backend = e.m.doctor()
    ranges = e.m.read(e.buffers["meshClusterRanges"], np.uint32).reshape(-1, 4)
    info = e.m.read(e.buffers["meshClusterInfo"], np.uint32).reshape(-1, 4)
    frontier_buffers = {
        getattr(e, "mesh_frontier_count", 32): e.buffers["meshFrontier"]
    }
    for count in {v.get("frontiers", 32) for v in f["variants"].values()} - set(
        frontier_buffers
    ):
        entries = []
        for first, *_ in info:
            pending = [int(first)]
            while len(pending) < count:
                splittable = [n for n in pending if ranges[n, 1] == 0]
                if not splittable:
                    break
                node = max(splittable, key=lambda n: int(ranges[n, 2]) - n)
                left = node + 1
                right = int(ranges[left, 2])
                at = pending.index(node)
                pending[at : at + 1] = [left, right]
            entries.extend([[n, int(ranges[n, 2]), 0, 0] for n in pending])
            entries.extend([[0, 0, 0, 0]] * (count - len(pending)))
        frontier_buffers[count] = e.m.buffer(np.asarray(entries, np.uint32))
    from sand_atlas_dem.gpu import Scan

    adjacency = {
        name: e.m.buffer(
            count=(2 * e.contact_capacity if name == "adjEntries" else e.n)
        )
        for name in ["adjCounts", "adjOffsets", "adjCursor", "adjEntries"]
    }
    scan = Scan(e.m, e.n)
    mode = "baseline"

    def dispatch(self, name, n, encoder, bindings):
        module = snapshot / f["variants"][mode]["shader"]
        values = (
            dict(bindings, **adjacency) if mode.startswith("segmented") else bindings
        )
        if name == "gather" and mode.startswith("segmented"):
            scan.encode(encoder, adjacency["adjCounts"], adjacency["adjOffsets"])
            self.m.kernel("scatterAdjacency", str(module)).dispatch(
                thread_count=[self.contact_capacity, 1, 1],
                command_encoder=encoder,
                vars={"r": values},
            )
            n = 64 * self.n
        self.m.kernel(name, str(module)).dispatch(
            thread_count=[max(1, int(n)), 1, 1],
            command_encoder=encoder,
            vars={"r": values},
        )

    e.dispatch = types.MethodType(dispatch, e)

    def reset():
        e.buffers["meshFrontier"] = frontier_buffers[
            f["variants"][mode].get("frontiers", 32)
        ]
        e.restore(saved, step)
        e.advance(p["warmup_steps"], rate=p["rate_s"], drag=p["drag_s"])

    records, checks, states = [], [], {}
    modes = p["modes"]
    # Forward/reverse order each repetition, six samples per mode.
    for repeat in range(p["repetitions"]):
        for order in [modes, list(reversed(modes))]:
            for mode in order:
                reset()
                start = time.perf_counter()
                e.advance(p["steps"], rate=p["rate_s"], drag=p["drag_s"])
                seconds = (time.perf_counter() - start) / p["steps"]
                current = e.state()
                if mode not in states:
                    states[mode] = current
                else:
                    checks.append(dict(mode=mode, **compare(states[mode], current, p)))
                records.append(dict(repeat=repeat, mode=mode, seconds_per_step=seconds))
                write_json(out / "samples.json", records)
    for index, reference in enumerate(modes):
        for mode in modes[index + 1 :]:
            checks.append(
                dict(
                    mode=mode,
                    reference=reference,
                    **compare(states[reference], states[mode], p),
                )
            )
    counters, stages = {}, []
    median = {
        m: statistics.median(r["seconds_per_step"] for r in records if r["mode"] == m)
        for m in modes
    }
    speedups = {m: median["baseline"] / v for m, v in median.items()}
    passed = all(c["passed"] for c in checks)
    write_json(
        out / "report.json",
        dict(
            identity=p["identity"],
            completed=True,
            backend=backend,
            sources_and_inputs_unchanged=intact(),
            median_seconds_per_step=median,
            speedups=speedups,
            eligible={
                m: passed and v >= p["minimum_speedup"]
                for m, v in speedups.items()
                if m != "baseline"
            },
            checks=checks,
            equivalence_passed=passed,
            counters=counters,
            scientific_admission=False,
        ),
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
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    p = json.loads(args.protocol.read_text())
    write_json(out / "protocol.json", p)
    paths = (
        list((root / "src").rglob("*.py"))
        + list((root / "src").rglob("*.slang"))
        + [
            Path(__file__).resolve(),
            root / "examples/benchmark_parallel_mesh.py",
            root / "examples/profile_runtime.py",
            root / "examples/segmented_shader.py",
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
    modes = variants(source)
    mode_records = {}
    for name, (text, pool) in modes.items():
        path = Path("sand_atlas_dem/shaders") / f"{name}.slang"
        dest = out / "source_snapshot" / path
        dest.write_text(text)
        hashes[str(path)] = sha256(dest)
        frontiers = (
            int(name.split("-")[0].removeprefix("frontier"))
            if name.startswith("frontier")
            else 32
        )
        mode_records[name] = dict(shader=str(path), pool=pool, frontiers=frontiers)
    inputs = {p[k]: sha256(p[k]) for k in ["checkpoint", "shape_library"] if k in p}
    for key in ["checkpoint", "shape"]:
        if key + "_sha256" in p:
            assert (
                inputs[p["checkpoint" if key == "checkpoint" else "shape_library"]]
                == p[key + "_sha256"]
            )
    write_json(
        out / "prospective-freeze.json",
        dict(
            sources=hashes,
            inputs=inputs,
            variants=mode_records,
            protocol_sha256=sha256(out / "protocol.json"),
        ),
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
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
