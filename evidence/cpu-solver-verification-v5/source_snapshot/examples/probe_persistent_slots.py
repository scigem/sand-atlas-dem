"""Independent synthetic endpoint-set verification of the GPU canonical-order cache."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.gpu import Engine
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
        / "source_snapshot/sand_atlas_dem/shaders/segmented-ordered-persistent-cached.slang"
    ).resolve()
    a = [(0, 1, 3, 0), (0, 1, 1, 0), (1, 2, 7, 0), (2, 0, 9, 0)]
    b = [(0, 1, 3, 0), (0, 1, 42, 0), (1, 2, 7, 0), (2, 0, 9, 0)]
    c = b[:2] + b[3:]
    d = c + [(0, 2, 66, 0)]
    full = [(0, 1, i, 0) for i in range(8192)]
    cases = [
        ("first", a),
        ("index-permutation", a[::-1]),
        ("equal-count-replacement", b),
        ("remove", c),
        ("add", d),
        ("permuted", d[::-1]),
        ("empty", []),
        ("reappear", a[::-1]),
        ("full8192", full),
        ("full8192-cache-hit", full[::-1]),
        ("reject8193", full + [(0, 1, 8192, 0)]),
    ]
    protocol = dict(
        identity=out.name,
        module=str(module),
        module_sha256=sha256(module),
        driver_sha256=sha256(__file__),
        cases=[
            dict(
                name=n,
                contacts=len(keys),
                keys_sha256=__import__("hashlib")
                .sha256(np.array(keys, np.uint32).tobytes())
                .hexdigest(),
            )
            for n, keys in cases
        ],
        gates="Exact canonical endpoint sequence, exact cache hit/miss counts from prior/current set equality, no duplicates, complete membership.8192 endpoints supported;8193 rejects with bit16384 before access. No force simulation or numerical-gate revision.",
    )
    write_json(out / "protocol.json", protocol)
    (out / "driver.py").write_bytes(Path(__file__).read_bytes())
    shape = from_mesh(trimesh.creation.box(), unit="mm", resolution=8, samples=32)
    m = Metal()
    e = Engine(
        ShapeLibrary([shape]),
        Experiment(
            particles=3,
            contact_geometry="mesh",
            timestep=1e-9,
            contacts_per_particle=4096,
        ),
        [[0.005] * 3, [0.01] * 3, [0.015] * 3],
        [0.03] * 3,
        [0] * 3,
        [1] * 3,
        metal=m,
    )
    r = e.bindings()
    r.update(
        {
            name: m.buffer(
                count=(
                    max(8192 * e.n, 2 * e.contact_capacity)
                    if name == "adjEntries"
                    else e.n
                )
            )
            for name in ["adjCounts", "adjOffsets", "adjCursor", "adjEntries"]
        }
    )
    r.update(
        adjCacheKeys=m.buffer(count=8192 * e.n, stride=16),
        adjCacheCounts=m.buffer(np.full(e.n, 0xFFFFFFFF, np.uint32)),
        adjCacheMetrics=m.buffer(np.zeros(2, np.uint32)),
    )
    r["slotNeedsBuild"] = m.buffer(count=e.n)
    previous = None
    records = []
    for name, keys in cases:
        meta = np.array(keys, np.uint32).reshape(-1, 4)
        expected = [
            sorted(
                [
                    tuple(k) + (endpoint,)
                    for k in keys
                    for endpoint, body in [(0, k[0]), (1, k[1])]
                    if body == i
                ]
            )
            for i in range(e.n)
        ]
        counts = np.array([len(x) for x in expected], np.uint32)
        ctrl = np.zeros(16, np.uint32)
        ctrl[2] = len(meta)
        r["control"].copy_from_numpy(ctrl)
        r["hash"].copy_from_numpy(np.full(e.hash_size, 0xFFFFFFFF, np.uint32))
        if len(meta):
            m.write_prefix(r["meta"], meta)
        r["adjCounts"].copy_from_numpy(counts)
        r["adjCursor"].copy_from_numpy(np.zeros(e.n, np.uint32))
        before = m.read(r["adjCacheMetrics"], np.uint32).astype(np.int64)
        encoder = m.device.create_command_encoder()
        for kernel, n in [("buildHistoryHash", e.contact_capacity)]:
            m.kernel(kernel, str(module)).dispatch(
                thread_count=[n, 1, 1], command_encoder=encoder, vars={"r": r}
            )
        m.kernel("validateSlots", str(module)).dispatch(
            thread_count=[64 * e.n, 1, 1], command_encoder=encoder, vars={"r": r}
        )
        for kernel, n in [
            ("scatterAdjacency", e.contact_capacity),
            ("orderAdjacency", 64 * e.n),
        ]:
            m.kernel(kernel, str(module)).dispatch(
                thread_count=[n, 1, 1], command_encoder=encoder, vars={"r": r}
            )
        m.submit(encoder)
        m.device.wait()
        flags = int(m.read(r["control"], np.uint32)[0])
        if name == "reject8193":
            assert flags == 16384
            records.append(dict(name=name, passed=True, flags=flags))
            continue
        assert flags == 0
        offsets = 8192 * np.arange(e.n, dtype=np.uint32)
        entries = m.read(r["adjEntries"], np.uint32)
        for body in range(e.n):
            active = entries[offsets[body] : offsets[body] + counts[body]]
            actual = [tuple(meta[x // 2]) + (int(x & 1),) for x in active]
            assert actual == expected[body], (name, body)
        metrics = (
            m.read(r["adjCacheMetrics"], np.uint32).astype(np.int64) - before
        ).tolist()
        hits = sum(
            previous is not None and previous[i] == expected[i] for i in range(e.n)
        )
        assert metrics == [hits, e.n - hits], (name, metrics, hits)
        records.append(
            dict(name=name, passed=True, cache_hit_miss=metrics, counts=counts.tolist())
        )
        previous = expected
        write_json(out / "checks.json", records)
    write_json(
        out / "report.json",
        dict(
            passed=all(r["passed"] for r in records),
            checks=records,
            backend=m.doctor(),
            module_unchanged=sha256(module) == protocol["module_sha256"],
            scientific_admission=False,
        ),
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
