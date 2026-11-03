"""Frozen brick completeness and exact distance probes, serial Metal only."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh
from brick_mesh import brick_shader, build_bricks
from coherent_mesh_shader import instrument

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.gpu import Engine, f4
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    asset = root / "assets/generated/concave-prism.h5"
    source = (root / "src/sand_atlas_dem/shaders/solver.slang").read_text()
    modules = {}
    for name, text in {
        "baseline": instrument(source),
        "brick": brick_shader(source, metrics=True),
    }.items():
        path = out / (name + ".slang")
        path.write_text(text)
        modules[name] = path
    files = {str(asset): sha256(asset)}
    for path in [
        Path(__file__),
        root / "examples/brick_mesh.py",
        root / "examples/coherent_mesh_shader.py",
    ]:
        shutil.copyfile(path, out / path.name)
        files[str(path)] = sha256(path)
    files.update({str(p): sha256(p) for p in modules.values()})
    policy = {
        "seed": 791,
        "random_points": 2048,
        "resolution": 32,
        "capacities": [96, 1],
        "distance_atol_m": 2e-9,
        "native_max_abs": 0.0,
        "complete_atol": 1e-12,
        "files": files,
        "scope": "Independent FP64 all-triangle candidate completeness; box analytical and concave triangle distances; exact GPU normal/distance comparison; cell boundaries, ties, external points, shape offsets and forced-small-cap fallback. Freeze before compilation; no CPU DEM.",
    }
    write_json(out / "protocol.json", policy)
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
    roots = e.m.read(e.buffers["meshRoots"], np.uint32).reshape(-1, 4)
    links = e.m.read(e.buffers["meshLinks"], np.uint32).reshape(-1, 4)
    triangles = e.m.read(e.buffers["meshTriangles"]).reshape(-1, 3, 4)
    records, receipts = [], []
    rng = np.random.default_rng(policy["seed"])
    points_by_shape = []
    for sid, shape in enumerate([box, concave]):
        lo = shape.vertices.min(0) / e.length_unit
        hi = shape.vertices.max(0) / e.length_unit
        points = rng.uniform(lo * 1.2, hi * 1.2, (policy["random_points"], 3)).astype(
            np.float32
        )
        # Rounded grid boundaries and their immediate float32 neighbors.
        cells = rng.integers(0, 33, (256, 3))
        boundary = (lo + cells * (hi - lo) / 32).astype(np.float32)
        points = np.vstack(
            [
                points,
                boundary,
                np.nextafter(boundary, np.float32(-np.inf)),
                np.nextafter(boundary, np.float32(np.inf)),
                np.zeros((1, 3), np.float32),
                triangles[int(links[int(roots[sid, 0]), 1]), :, :3],
            ]
        )
        points_by_shape.append(points)
    for capacity in policy["capacities"]:
        arrays, receipt = build_bricks(
            roots, links, triangles, resolution=policy["resolution"], capacity=capacity
        )
        receipts.append(receipt)
        write_json(out / "brick-builds.json", receipts)
        buffers = {k: e.m.buffer(a) for k, a in arrays.items()}
        for sid, (shape, points) in enumerate(zip([box, concave], points_by_shape)):
            # Exhaustive nearest triangle reference uses rounded GPU geometry.
            leaves = links[int(roots[sid, 0]) : int(roots[sid, 1])]
            leaves = leaves[leaves[:, 2] > 0]
            first = int(leaves[:, 1].min())
            end = int((leaves[:, 1] + leaves[:, 2]).max())
            tri = triangles[first:end, :, :3].astype(float)
            complete = True
            covered = 0
            for p in points:
                g = (p - arrays["brickOrigin"][sid, :3]) * arrays["brickScale"][sid, :3]
                if np.any(g < 0) or np.any(g >= 32):
                    continue
                cell = np.floor(g).astype(int)
                bid = (
                    int(arrays["brickInfo"][sid, 0])
                    + (cell[0] * 32 + cell[1]) * 32
                    + cell[2]
                )
                offset, count = map(int, arrays["brickRanges"][bid, :2])
                if not count:
                    continue
                if (
                    np.linalg.norm(p - arrays["brickCenters"][bid, :3])
                    > arrays["brickCenters"][bid, 3]
                ):
                    continue
                closest = trimesh.triangles.closest_point(
                    tri, np.repeat(p[None, :], len(tri), axis=0)
                )
                dist = np.linalg.norm(closest - p, axis=1)
                winning = (
                    np.flatnonzero(dist <= dist.min() + policy["complete_atol"]) + first
                )
                available = arrays["brickTriangles"][offset : offset + count]
                complete &= bool(np.all(np.isin(winning, available)))
                covered += 1
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
            result = e.m.buffer(count=len(points), stride=16)
            actual = {}
            counts = {}
            for mode in ["baseline", "brick"]:
                metrics = e.m.buffer(np.zeros(9, np.uint32))
                bindings = dict(e.bindings(), meshQueryMetrics=metrics)
                if mode == "brick":
                    bindings.update(buffers)
                e.m.kernel("meshProbe", str(modules[mode])).dispatch(
                    thread_count=[len(points), 1, 1],
                    vars={"r": bindings},
                    count=len(points),
                    shape=sid,
                    points=a,
                    result=result,
                )
                e.m.device.wait()
                actual[mode] = e.m.read(result).reshape(-1, 4).copy()
                counts[mode] = e.m.read(metrics, np.uint32).tolist()
            delta = float(np.max(abs(actual["brick"] - actual["baseline"])))
            error = float(np.max(abs(actual["brick"][:, 3] * e.length_unit - expected)))
            record = {
                "shape": sid,
                "capacity": capacity,
                "points": len(points),
                "cpu_candidate_queries": covered,
                "complete": complete,
                "native_max_abs": delta,
                "independent_distance_max_abs_m": error,
                "metrics": counts,
                "passed": complete
                and delta <= policy["native_max_abs"]
                and error <= policy["distance_atol_m"],
            }
            records.append(record)
            np.savez(
                out / f"shape-{sid}-cap-{capacity}.npz",
                points=points,
                expected=expected,
                **actual,
            )
    intact = all(sha256(p) == h for p, h in files.items())
    fast = sum(r["metrics"]["brick"][5] for r in records if r["capacity"] == 96)
    fallback = sum(r["metrics"]["brick"][0] for r in records if r["capacity"] == 1)
    ties = sum(r["metrics"]["brick"][8] for r in records if r["capacity"] == 96)
    passed = (
        intact
        and all(r["passed"] for r in records)
        and fast > 0
        and fallback > 0
        and ties > 0
    )
    write_json(
        out / "report.json",
        {
            "checks": records,
            "passed": passed,
            "sources_unchanged": intact,
            "path_coverage": {
                "fast": fast,
                "forced_small_cap_fallback": fallback,
                "near_tie_fallback": ties,
            },
            "backend": e.m.doctor(),
            "scientific_admission": False,
        },
    )
    assert passed


if __name__ == "__main__":
    main()
