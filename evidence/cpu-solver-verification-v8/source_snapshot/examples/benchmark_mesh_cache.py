"""Prospective exact-array cache benchmark; raw geometry stays private."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sand_atlas_dem.util import sha256, write_json


def execute(out):
    from sand_atlas_dem.geometry import ShapeLibrary
    from sand_atlas_dem.mesh_cache import array_hash, packed_meshes

    policy = json.loads((out / "protocol.json").read_text())
    freeze = json.loads((out / "prospective-freeze.json").read_text())
    snapshot = out / "source_snapshot"
    assert all(sha256(snapshot / p) == h for p, h in freeze["sources"].items())
    assert sha256(policy["shape_library"]) == policy["shape_sha256"]
    assert sha256(out / "protocol.json") == freeze["protocol_sha256"]
    start = time.perf_counter()
    library = ShapeLibrary.load(policy["shape_library"])
    load_seconds = time.perf_counter() - start
    samples = []
    expected = None
    for i in range(1 + policy["warm_repetitions"]):
        start = time.perf_counter()
        arrays, receipt = packed_meshes(
            library, policy["length_unit"], directory=out / "private/cache"
        )
        seconds = time.perf_counter() - start
        hashes = {n: array_hash(a) for n, a in arrays.items()}
        if expected is None:
            expected = hashes
        assert hashes == expected
        assert receipt["status"] == ("miss" if i == 0 else "hit")
        samples.append(dict(seconds=seconds, receipt=receipt, array_hashes=hashes))
        del arrays
        write_json(out / "samples.json", samples)
    import statistics

    warm = statistics.median(s["seconds"] for s in samples[1:])
    speedup = samples[0]["seconds"] / warm
    unchanged = all(sha256(snapshot / p) == h for p, h in freeze["sources"].items())
    unchanged &= sha256(policy["shape_library"]) == policy["shape_sha256"]
    write_json(
        out / "report.json",
        dict(
            completed=True,
            library_load_seconds=load_seconds,
            cold_pack_and_write_seconds=samples[0]["seconds"],
            warm_median_seconds=warm,
            speedup=speedup,
            performance_passed=speedup >= policy["minimum_speedup"],
            exact_arrays=True,
            sources_and_inputs_unchanged=unchanged,
            scope="CPU geometry preprocessing only, not a CPU DEM fallback. Includes input hashing, all cached-array integrity checks and disk reads/writes; excludes HDF5 library load and device upload. OS filesystem cache is not flushed.",
            scientific_admission=False,
        ),
    )
    assert unchanged and speedup >= policy["minimum_speedup"]


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
    policy = json.loads(args.protocol.read_text())
    assert sha256(policy["shape_library"]) == policy["shape_sha256"]
    write_json(out / "protocol.json", policy)
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "src/sand_atlas_dem").rglob("*.py")) + list(
        (root / "src/sand_atlas_dem").rglob("*.slang")
    )
    sources = {}
    for p in paths + [Path(__file__).resolve()]:
        relative = (
            p.relative_to(root / "src")
            if p.is_relative_to(root / "src")
            else Path(p.name)
        )
        dest = out / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        sources[str(relative)] = sha256(p)
    write_json(
        out / "prospective-freeze.json",
        dict(sources=sources, protocol_sha256=sha256(out / "protocol.json")),
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
