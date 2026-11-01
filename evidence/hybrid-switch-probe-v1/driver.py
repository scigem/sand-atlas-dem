"""Frozen synthetic GPU dispatch transitions with independent FP64 force/torque sums."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.gpu import Engine, Scan
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    module = (
        args.study
        / "source_snapshot/sand_atlas_dem/shaders/segmented-ordered-cached-hybrid.slang"
    ).resolve()
    sizes = [191, 192, 193, 191, 0, 192, 192]
    p = dict(
        identity=out.name,
        driver_sha256=sha256(__file__),
        module=str(module),
        module_sha256=sha256(module),
        contact_counts=sizes,
        threshold=192,
        force_torque_rtol=2e-6,
        force_torque_atol=1e-10,
        scope="Prescribed synthetic contact rows verify GPU sparse/dense transitions and reductions; no CPU DEM simulation. Exact full endpoint membership and expected order-kernel activity required. Independent FP64 sums from actual FP32 GPU contact forces and prescribed arms, zero momentum and physical history step0. No scientific admission.",
    )
    write_json(out / "protocol.json", p)
    (out / "driver.py").write_bytes(Path(__file__).read_bytes())
    shape = from_mesh(trimesh.creation.box(), unit="mm", resolution=8, samples=32)
    m = Metal()
    e = Engine(
        ShapeLibrary([shape]),
        Experiment(
            particles=3,
            contact_geometry="mesh",
            periodic=False,
            timestep=1e-9,
            contacts_per_particle=4096,
        ),
        [[0.005] * 3, [0.01] * 3, [0.015] * 3],
        [0.03] * 3,
        [0] * 3,
        [1] * 3,
        metal=m,
    )
    r = e.bindings(dt=0)
    r.update(
        {
            name: m.buffer(
                count=(2 * e.contact_capacity if name == "adjEntries" else e.n)
            )
            for name in ["adjCounts", "adjOffsets", "adjCursor", "adjEntries"]
        }
    )
    r.update(
        adjCacheKeys=m.buffer(count=8192 * e.n, stride=16),
        adjCacheCounts=m.buffer(np.full(e.n, 0xFFFFFFFF, np.uint32)),
        adjCacheMetrics=m.buffer(np.zeros(2, np.uint32)),
    )
    scan = Scan(m, e.n)
    records = []
    for size in sizes:
        meta = np.array([[0, 1, i, 0] for i in range(size)], np.uint32).reshape(-1, 4)
        ctrl = np.zeros(16, np.uint32)
        ctrl[2] = size
        r["control"].copy_from_numpy(ctrl)
        pd = np.tile(np.array([-0.4, 0.01, 0, 0.001], np.float32), (size, 1))
        if size:
            m.write_prefix(r["meta"], meta)
            m.write_prefix(r["pointDepth"], pd)
            m.write_prefix(
                r["normalArea"],
                np.tile(np.array([-1, 0, 0, 1e-6], np.float32), (size, 1)),
            )
        before = m.read(r["adjCacheMetrics"], np.uint32).astype(np.int64)
        enc = m.device.create_command_encoder()
        for name, n in [
            ("clearContactStorage", max(e.hash_size, e.n)),
            ("resolveContacts", e.contact_capacity),
            ("buildHistoryHash", e.contact_capacity),
        ]:
            m.kernel(name, str(module)).dispatch(
                thread_count=[n, 1, 1], command_encoder=enc, vars={"r": r}
            )
        scan.encode(enc, r["adjCounts"], r["adjOffsets"])
        for name, n in [
            ("scatterAdjacency", e.contact_capacity),
            ("orderAdjacency", 64 * e.n),
            ("gather", 64 * e.n),
        ]:
            m.kernel(name, str(module)).dispatch(
                thread_count=[n, 1, 1], command_encoder=enc, vars={"r": r}
            )
        m.submit(enc)
        m.device.wait()
        flags = int(m.read(r["control"], np.uint32)[0])
        assert flags == 0
        dense = size >= 192
        counts = m.read(r["adjCounts"], np.uint32)
        offsets = m.read(r["adjOffsets"], np.uint32)
        entries = m.read(r["adjEntries"], np.uint32)
        heads = m.read(r["heads"], np.uint32)
        links = m.read(r["next"], np.uint32)
        for body in range(e.n):
            expected = [
                2 * i + endpoint
                for i, k in enumerate(meta)
                for endpoint, owner in [(0, k[0]), (1, k[1])]
                if owner == body
            ]
            if dense:
                actual = entries[offsets[body] : offsets[body] + counts[body]].tolist()
                assert actual == expected
            else:
                actual = []
                head = heads[body]
                while int(head) != 0xFFFFFFFF:
                    assert len(actual) <= 2 * e.contact_capacity
                    actual.append(int(head))
                    head = links[head]
                assert sorted(actual) == expected
        metrics = (
            m.read(r["adjCacheMetrics"], np.uint32).astype(np.int64) - before
        ).tolist()
        assert sum(metrics) == (3 if dense else 0)
        cf = m.read_prefix(r["contactForce"], 4 * size).reshape(-1, 4).astype(float)
        position = m.read(r["position"]).reshape(-1, 4).astype(float)
        branch = position[0, :3] - position[1, :3]
        expected_force = np.zeros((3, 3))
        expected_torque = np.zeros((3, 3))
        if size:
            expected_force[0] = cf[:, :3].sum(0)
            expected_force[1] = -expected_force[0]
            expected_torque[0] = np.cross(pd[:, :3].astype(float), cf[:, :3]).sum(0)
            expected_torque[1] = np.cross(
                pd[:, :3].astype(float) - branch, -cf[:, :3]
            ).sum(0)
        force = m.read(r["force"]).reshape(-1, 4)[:, :3]
        torque = m.read(r["torque"]).reshape(-1, 4)[:, :3]
        assert np.allclose(
            force,
            expected_force,
            rtol=p["force_torque_rtol"],
            atol=p["force_torque_atol"],
        )
        assert np.allclose(
            torque,
            expected_torque,
            rtol=p["force_torque_rtol"],
            atol=p["force_torque_atol"],
        )
        records.append(
            dict(
                contacts=size,
                dense=dense,
                cache_hit_miss=metrics,
                passed=True,
                force_max_abs=float(abs(force - expected_force).max()),
                torque_max_abs=float(abs(torque - expected_torque).max()),
            )
        )
        write_json(out / "checks.json", records)
    write_json(
        out / "report.json",
        dict(
            passed=True,
            checks=records,
            module_unchanged=sha256(module) == p["module_sha256"],
            backend=m.doctor(),
            scientific_admission=False,
        ),
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
