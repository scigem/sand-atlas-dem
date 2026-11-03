"""CPU-only static geometry diagnostic of checkpoint active-contact bricks."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from sand_atlas_dem.util import sha256, write_json


def rotate(q, p):
    return p + 2 * np.cross(q[:3], np.cross(q[:3], p) + q[3] * p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    p = json.loads(args.protocol.read_text())
    files = {
        str(path): sha256(path)
        for path in [
            args.protocol,
            Path(__file__),
            Path(p["checkpoint"]),
            Path(p["shape_library"]),
        ]
    }
    write_json(
        out / "protocol.json",
        {
            "files": files,
            "scope": "Static FP64 checkpoint contact-coordinate diagnostic; counts conservative brick candidates directly. Not a CPU simulation or a throughput measurement.",
        },
    )
    (out / "driver.py").write_bytes(Path(__file__).read_bytes())
    with h5py.File(p["checkpoint"]) as h:
        spec = json.loads(h.attrs["experiment"])
        meta = h["meta"][:]
        positions = h["position"][:, :3].astype(float) + h["positionRemainder"][:, :3]
        rotation = h["rotation"][:].astype(float)
        cell = h["cell"][0, :3]
    with h5py.File(p["shape_library"]) as h:
        count = len(h)
        length = float(
            np.median(
                [
                    (6 * h[str(i)].attrs["volume"] / np.pi) ** (1 / 3)
                    for i in range(count)
                ]
            )
        )
        rng = np.random.default_rng(spec["seed"])
        ids = (
            rng.permutation(np.arange(spec["particles"], dtype=np.uint32) % count)
            if spec["shape_sampling"] == "balanced"
            else rng.integers(count, size=spec["particles"], dtype=np.uint32)
        )
        samples = {}
        for sid in np.unique(ids[meta[:, 0]]):
            nodes = np.unique(meta[ids[meta[:, 0]] == sid, 2])
            values = h[str(sid)]["samples"][nodes]
            samples.update(
                {
                    (int(sid), int(node)): value / length
                    for node, value in zip(nodes, values)
                }
            )
        shapes = {}
        for sid in np.unique(ids[meta[:, 1]]):
            group = h[str(sid)]
            vertices = (group["vertices"][:] / length).astype(np.float32)
            tri = vertices[group["faces"][:]].astype(float)
            low = tri.min(axis=(0, 1)).astype(np.float32)
            step = ((tri.max(axis=(0, 1)) - low) / p["brick_resolution"]).astype(
                np.float32
            )
            margin = 128 * np.finfo(np.float32).eps * (1 + abs(tri).max())
            radius = np.nextafter(
                np.float32(np.linalg.norm(step.astype(float) * 0.5) + margin),
                np.float32(np.inf),
            )
            shapes[int(sid)] = (
                tri,
                low,
                step,
                margin,
                radius,
                cKDTree(tri.reshape(-1, 3)),
            )
        rows = []
        for src, dst, node, _ in meta:
            src, dst, node = map(int, (src, dst, node))
            branch = positions[dst] - positions[src]
            if spec["periodic"]:
                branch -= np.floor(branch / cell + 0.5) * cell
            arm = rotate(rotation[src], samples[(int(ids[src]), node)] * spec["scale"])
            inv = rotation[dst].copy()
            inv[:3] *= -1
            local = rotate(inv, arm - branch) / spec["scale"]
            sid = int(ids[dst])
            tri, low, step, margin, radius, tree = shapes[sid]
            cellid = np.floor((local - low) * (1 / step).astype(np.float32)).astype(int)
            outside = bool(
                np.any(cellid < 0) or np.any(cellid >= p["brick_resolution"])
            )
            center = (low.astype(float) + (cellid + 0.5) * step.astype(float)).astype(
                np.float32
            )
            sphere = bool(np.linalg.norm(local - center) <= radius)
            upper = float(tree.query(center.astype(float))[0])
            threshold = upper + 2 * float(radius) + margin
            d = np.maximum(
                np.maximum(tri.min(axis=1) - center, center - tri.max(axis=1)), 0
            )
            candidates = int(
                np.count_nonzero(np.einsum("ij,ij->i", d, d) <= threshold**2)
            )
            rows.append(
                {
                    "target_shape": sid,
                    "outside": outside,
                    "inside_sphere": sphere,
                    "candidates": candidates,
                }
            )
    intact = all(sha256(path) == digest for path, digest in files.items())
    write_json(
        out / "report.json",
        {
            "contacts": len(rows),
            "outside_directory": sum(r["outside"] for r in rows),
            "outside_sphere": sum(not r["inside_sphere"] for r in rows),
            "over_capacity": sum(r["candidates"] > p["brick_capacity"] for r in rows),
            "candidate_min": min(r["candidates"] for r in rows),
            "candidate_max": max(r["candidates"] for r in rows),
            "sources_unchanged": intact,
            "rows": rows,
        },
    )
    assert intact


if __name__ == "__main__":
    main()
