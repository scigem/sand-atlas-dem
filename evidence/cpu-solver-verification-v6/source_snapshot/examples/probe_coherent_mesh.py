"""Frozen exact-query coherence checks with independent FP64 distance oracles."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.gpu import Engine, f4
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import sha256, write_json

PROBE = """
[shader("compute")][numthreads(64,1,1)]
void coherentProbe(uint id:SV_DispatchThreadID,uniform uint count,uniform uint shape,
                  uniform uint first,uniform uint triangles,uniform uint rounds,
                  StructuredBuffer<float4> points,RWStructuredBuffer<float4> result){
    if(id>=count)return;
    uint seed=id%2==0?0xffffffffu:first+(id%triangles);
    for(uint step=0;step<rounds;step++){
        uint query=(id+step*137)%count;
        result[step*count+id]=meshDistanceCoherent(shape,points[query].xyz,seed);
    }
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    modules = {}
    for name in ["baseline", "coherent"]:
        original = (
            args.study / "source_snapshot/sand_atlas_dem/shaders" / (name + ".slang")
        )
        path = out / (name + ".slang")
        path.write_text(original.read_text() + (PROBE if name == "coherent" else ""))
        modules[name] = path
    asset = Path("assets/generated/concave-prism.h5").resolve()
    protocol = {
        "identity": out.name,
        "seed": 791,
        "count": 1024,
        "rounds": 4,
        "distance_atol_m": 2e-9,
        "native_atol": 0.0,
        "modules": {str(p): sha256(p) for p in modules.values()},
        "asset": {str(asset): sha256(asset)},
        "driver_sha256": sha256(__file__),
        "scope": "Random near/interior/exterior points; alternating cold and arbitrary valid triangle seeds; repeated queries jump across geometry; packed shape offsets. Independent FP64 analytical box and trimesh triangle distance. No CPU DEM.",
    }
    write_json(out / "protocol.json", protocol)
    shutil.copyfile(__file__, out / "driver.py")
    box = from_mesh(
        trimesh.creation.box(extents=[1, 2, 3]), unit="mm", resolution=8, samples=32
    )
    concave = ShapeLibrary.load(asset).shapes[0]
    e = Engine(
        ShapeLibrary([box, concave]),
        Experiment(particles=2, timestep=1e-9, contact_geometry="mesh"),
        [[0.005] * 3, [0.015] * 3],
        [0.03] * 3,
        [0, 1],
        [1, 1],
        metal=Metal(),
    )
    links = e.m.read(e.buffers["meshLinks"], np.uint32).reshape(-1, 4)
    roots = e.m.read(e.buffers["meshRoots"], np.uint32).reshape(-1, 4)
    records = []
    rng = np.random.default_rng(protocol["seed"])
    for sid, shape in enumerate([box, concave]):
        points = rng.uniform(
            shape.vertices.min(0) * 1.2,
            shape.vertices.max(0) * 1.2,
            (protocol["count"], 3),
        )
        points = (points / e.length_unit).astype(np.float32)
        if sid == 0:
            q = abs(points.astype(float) * e.length_unit) - np.array(
                [0.0005, 0.001, 0.0015]
            )
            expected = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(
                q.max(axis=1), 0
            )
        else:
            mesh = trimesh.Trimesh(
                shape.vertices / e.length_unit, shape.faces, process=False
            )
            expected = (
                -trimesh.proximity.signed_distance(mesh, points.astype(float))
                * e.length_unit
            )
        a = e.m.buffer(f4(points))
        b = e.m.buffer(count=len(points), stride=16)
        e.m.kernel("meshProbe", str(modules["baseline"])).dispatch(
            thread_count=[len(points), 1, 1],
            vars={"r": e.bindings()},
            count=len(points),
            shape=sid,
            points=a,
            result=b,
        )
        e.m.device.wait()
        native = e.m.read(b).reshape(-1, 4)
        # Descendant links are contiguous until the root escape; leaf triangle ids
        # are globally packed and must stay in the selected shape's range.
        root = int(roots[sid, 0])
        leaves = links[root : int(links[root, 0])]
        leaves = leaves[leaves[:, 2] > 0]
        first = int(leaves[:, 1].min())
        triangles = int((leaves[:, 1] + leaves[:, 2]).max()) - first
        result = e.m.buffer(count=len(points) * protocol["rounds"], stride=16)
        e.m.kernel("coherentProbe", str(modules["coherent"])).dispatch(
            thread_count=[len(points), 1, 1],
            vars={"r": e.bindings()},
            count=len(points),
            shape=sid,
            first=first,
            triangles=triangles,
            rounds=protocol["rounds"],
            points=a,
            result=result,
        )
        e.m.device.wait()
        actual = e.m.read(result).reshape(protocol["rounds"], len(points), 4)
        for step in range(protocol["rounds"]):
            order = (np.arange(len(points)) + step * 137) % len(points)
            delta = float(np.max(abs(actual[step] - native[order])))
            error = float(
                np.max(abs(actual[step, :, 3] * e.length_unit - expected[order]))
            )
            records.append(
                {
                    "shape": sid,
                    "round": step,
                    "native_max_abs": delta,
                    "independent_distance_max_abs_m": error,
                    "passed": delta <= protocol["native_atol"]
                    and error <= protocol["distance_atol_m"],
                }
            )
        np.savez(
            out / f"shape-{sid}.npz",
            points=points,
            expected=expected,
            native=native,
            actual=actual,
        )
    intact = all(
        sha256(p) == h for key in ["modules", "asset"] for p, h in protocol[key].items()
    )
    write_json(
        out / "report.json",
        {
            "checks": records,
            "passed": intact and all(r["passed"] for r in records),
            "sources_unchanged": intact,
            "backend": e.m.doctor(),
            "scientific_admission": False,
        },
    )
    assert intact and all(r["passed"] for r in records)


if __name__ == "__main__":
    main()
