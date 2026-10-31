"""Independently compare physical HDF5 fields; no GPU or assembly admission."""

import json
from pathlib import Path

import h5py
import numpy as np

from sand_atlas_dem.util import read_json, sha256, write_json

out = Path(__file__).resolve().parent
receipt = read_json(out / "preparation-receipt.json")
parent = Path(
    "/Users/bmar5496/code/sand-atlas-dem/assets/private/hostun-30-r64-s2048.h5"
)
asset = Path(receipt["output"])
assert sha256(parent) == receipt["parent_sha256"] and sha256(asset) == receipt["sha256"]
levels = {r["shape"]: r["subdivisions"] for r in receipt["shapes"]}
counts = {}
with h5py.File(parent) as a, h5py.File(asset) as b:
    assert set(a) == set(b) and len(a) == 53
    for key in a:
        x, y = a[key], b[key]
        for name in ("vertices", "faces", "sdf", "origin", "inertia"):
            assert np.array_equal(x[name][:], y[name][:]), (key, name)
        for name in ("id", "volume", "spacing", "radius"):
            assert x.attrs[name] == y.attrs[name], (key, name)
        ident = str(y.attrs["id"])
        level = levels[ident]
        provenance = json.loads(y.attrs["provenance"])
        assert provenance["quadrature_subdivisions"] == level
        assert (
            not provenance["quadrature_gpu_verified"]
            and not provenance["material_qualified"]
        )
        assert y["samples"].shape == (3 * 4**level * len(x["faces"]), 3)
        counts[ident] = len(y["samples"])
freeze = read_json(out / "prospective-freeze.json")
root = out.parents[1]
assert all(
    sha256(root / name) == digest for name, digest in freeze["source_hashes"].items()
)
assert sha256(asset) == receipt["sha256"]
write_json(
    out / "verification.json",
    dict(
        passed=True,
        source_unchanged=True,
        all_53_physical_shapes_exactly_preserved=True,
        sample_counts=counts,
        total_samples=sum(counts.values()),
        asset_sha256=receipt["sha256"],
        selection_target=0.05,
        worst_selected_reference_error=max(
            r["worst_reference_relative_error"] for r in receipt["shapes"]
        ),
        gpu_executed=False,
        gpu_verified=False,
        scientific_admission=False,
    ),
)
