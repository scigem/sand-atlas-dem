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
    result = {"baseline": (source, 4096)}
    for pool in [256, 1024, 16384]:
        assert source.count("task+=4096") == 1
        result[f"pool{pool}"] = (source.replace("task+=4096", f"task+={pool}"), pool)
    # Stronger conservative root-only rejection: all hull-plane groups, using
    # the existing outward-padded corner box. It never replaces exact queries.
    start = source.index("bool meshClusterMayContact(")
    end = source.index('[shader("compute")]', start)
    helper = source[start:end].replace(
        "meshClusterMayContact", "meshRootAllPlanesMayContact"
    )
    helper = helper.replace(
        "    return meshBoxMayContact(r.shapeId[dst],low,high);",
        """    float3 center=.5*(low+high),half=.5*(high-low);
    for(uint group=0;group<6;group++){
        uint4 info=r.meshHullInfo[6*r.shapeId[dst]+group];
        for(uint n=0;n<info.y;n++){
            float4 plane=r.meshPlanes[info.x+n];
            if(dot(plane.xyz,center)+plane.w-dot(abs(plane.xyz),half)>0)return false;
        }
    }
    return true;""",
    )
    stronger = source[:start] + helper + source[start:]
    for a, b, bit in [("ij.x", "ij.y", 1), ("ij.y", "ij.x", 2)]:
        old = f"if(meshClusterMayContact({a},{b},r.meshClusterInfo[r.shapeId[{a}]].x))mask|={bit};"
        assert stronger.count(old) == 1
        stronger = stronger.replace(
            old,
            old.replace(
                "))mask",
                f")&&meshRootAllPlanesMayContact({a},{b},r.meshClusterInfo[r.shapeId[{a}]].x))mask",
            ),
        )
    result["all-hull-roots"] = (stronger, 4096)
    for lanes in [1, 4, 8, 16, 32]:
        old = '[shader("compute")][numthreads(64,1,1)]\nvoid gather('
        assert source.count(old) == 1
        result[f"gather{lanes}"] = (
            source.replace(
                old, old.replace("numthreads(64", "numthreads(" + str(lanes))
            ),
            4096,
        )
    for frontier, pool in [
        (8, 4096),
        (16, 4096),
        (64, 4096),
        (128, 4096),
        (64, 16384),
        (128, 16384),
    ]:
        text = source.replace(
            "r.meshFrontier[32*r.shapeId[i]+tile]",
            f"r.meshFrontier[{frontier}*r.shapeId[i]+tile]",
        )
        text = text.replace("uint tasks=64*min", f"uint tasks={2 * frontier}*min")
        text = text.replace(
            "task/64,(task%64)/32,task%32",
            f"task/{2 * frontier},(task%{2 * frontier})/{frontier},task%{frontier}",
        )
        text = text.replace(
            f"task%{frontier},lane,32", f"task%{frontier},lane,{frontier}"
        )
        text = text.replace("task+=4096", f"task+={pool}")
        result[f"frontier{frontier}-pool{pool}"] = (text, pool)
    return result


def instrument(source):
    source = source.replace(
        "    RWStructuredBuffer<uint> meshPairMask;",
        "    RWStructuredBuffer<uint> meshPairMask;\n    RWStructuredBuffer<uint> meshWork;",
    )
    pos = source.index("// Root rejection is shared")
    source = (
        source[:pos]
        + "void meshMetric(uint slot,uint count){uint unused;InterlockedAdd(r.meshWork[slot],count,unused);}\n"
        + source[pos:]
    )
    source = source.replace(
        "    r.meshPairMask[id]=mask;",
        "    r.meshPairMask[id]=mask;meshMetric(0,1);meshMetric(1,countbits(mask));",
    )
    source = source.replace(
        "    uint cluster=subtree.x;",
        "    if(lane==0)meshMetric(2,1);\n    uint cluster=subtree.x;",
    )
    source = source.replace(
        "            float3 low,high;meshPairBounds(2*pair+pass,cluster,low,high);",
        "            meshMetric(3,1);meshMetric(8+2*pair,1);\n            float3 low,high;meshPairBounds(2*pair+pass,cluster,low,high);",
    )
    source = source.replace(
        "        if(candidate!=0&&lane<range.y){",
        "        if(lane==0&&candidate!=0){meshMetric(4,1);meshMetric(5,range.y);meshMetric(9+2*pair,range.y);}\n        if(candidate!=0&&lane<range.y){",
    )
    # Only the subtree collector has this whitespace/layout.
    source = source.replace(
        "            if(nodeContact(i,j,node,pd,na)){",
        "            if(nodeContact(i,j,node,pd,na)){meshMetric(6,1);",
    )
    return source


def execute(out):
    from benchmark_parallel_mesh import compare
    from profile_runtime import split_step

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
    ordinary_dispatch = e.dispatch
    mode = "baseline"
    metrics = None

    def dispatch(self, name, n, encoder, bindings):
        if name in {
            "prepareMeshPairs",
            "collectMeshFrontiers",
            "collectMeshFineFrontiers",
        } or (name == "gather" and mode.startswith("gather")):
            pool = f["variants"][mode]["pool"]
            if name in {"collectMeshFrontiers", "collectMeshFineFrontiers"}:
                n = 64 * pool
                name = "collectMeshFrontiers"
            module = snapshot / f["variants"][mode]["shader"]
            values = (
                dict(bindings, meshWork=metrics)
                if mode.endswith("-metrics")
                else bindings
            )
            self.m.kernel(name, str(module)).dispatch(
                thread_count=[max(1, int(n)), 1, 1],
                command_encoder=encoder,
                vars={"r": values},
            )
        else:
            ordinary_dispatch(name, n, encoder, bindings)

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
    for mode in modes[1:]:
        checks.append(dict(mode=mode, **compare(states["baseline"], states[mode], p)))
    # New implementation profile: split wall timings are diagnostic, not GPU timestamps.
    mode = "baseline"
    reset()
    reference_start = e.state()
    e.advance(p["split_steps"], rate=p["rate_s"], drag=p["drag_s"])
    reference = e.state()
    e.restore(reference_start, e.step - p["split_steps"])
    stages = [split_step(e, p) for _ in range(p["split_steps"])]
    checks.append(dict(mode="split-profile", **compare(reference, e.state(), p)))
    write_json(out / "split-stages.json", stages)
    counters = {}
    for base in ["baseline", "all-hull-roots"]:
        mode = base + "-metrics"
        metrics = e.m.buffer(np.zeros(8 + 2 * e.pair_capacity, np.uint32))
        reset()
        metrics.copy_from_numpy(np.zeros(8 + 2 * e.pair_capacity, np.uint32))
        e.advance(1, rate=p["rate_s"], drag=p["drag_s"])
        work = e.m.read(metrics, np.uint32)
        pairs = int(work[0])
        per_pair = work[8 : 8 + 2 * pairs].reshape(-1, 2)
        counters[base] = dict(
            totals=work[:8].tolist(),
            per_pair_clusters_and_samples=per_pair.tolist(),
            contacts=e.state()["contacts"],
        )
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
            stage_mean_seconds={
                name: float(
                    np.mean(
                        [
                            s["wall_seconds"]
                            for row in stages
                            for s in row
                            if s["name"] == name
                        ]
                    )
                )
                for name in [s["name"] for s in stages[0]]
            },
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
    for name in ["baseline", "all-hull-roots"]:
        modes[name + "-metrics"] = (instrument(modes[name][0]), modes[name][1])
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
