"""Strict FP64 traction at unchanged source nodes and frozen physical poses."""

import argparse
import importlib.metadata
import shutil
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.reference import triangle_nearest_points
from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_hash = sha256(args.protocol)
    policy = read_json(args.protocol)
    for name in ("library", "pose_report"):
        if sha256(policy[name]) != policy[f"{name}_sha256"]:
            raise ValueError(f"Frozen {name} changed")
    if policy["nearest_point_policy"] != "strict-fp64-minimum-squared-distance":
        raise ValueError("Explicit strict-distance policy required")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), *sorted((root / "src/sand_atlas_dem").glob("*.py"))]
    frozen = {str(p.relative_to(root)): sha256(p) for p in paths}
    for path in paths:
        target = args.output / "source_snapshot" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(args.protocol, args.output / "protocol.json")
    write_json(
        args.output / "prospective-freeze.json",
        dict(source_hashes=frozen, protocol_sha256=protocol_hash),
    )
    library = ShapeLibrary.load(policy["library"])
    shapes = {shape.id: shape for shape in library.shapes}
    rows = []
    for case in read_json(policy["pose_report"])["rows"]:
        shape = shapes[case["shape"]]
        unit = (6 * shape.volume / np.pi) ** (1 / 3)
        mesh = trimesh.Trimesh(shape.vertices / unit, shape.faces, process=False)
        branch = np.array([case["separation_m"] / unit, 0, 0])
        force = np.zeros(3)
        counts = []
        maximum_depth = 0.0
        for sign in (1, -1):
            points = shape.samples / unit - sign * branch
            # A point outside the closed source mesh's box contributes zero.
            mask = np.all(
                (points >= mesh.bounds[0]) & (points <= mesh.bounds[1]), axis=1
            )
            points = points[mask]
            areas = shape.areas[mask]
            subtotal = np.zeros(3)
            active_count = 0
            for start in range(0, len(points), policy["query_batch_size"]):
                block = points[start : start + policy["query_batch_size"]]
                nearest, distance, _ = triangle_nearest_points(mesh, block)
                inside = mesh.contains(block)
                active_count += int(inside.sum())
                if inside.any():
                    maximum_depth = max(maximum_depth, float(distance[inside].max()))
                    subtotal += (
                        0.5
                        * np.sum(
                            areas[start : start + len(block)][inside, None]
                            * (nearest[inside] - block[inside]),
                            axis=0,
                        )
                        * unit
                    )
            force += sign * subtotal
            counts.append(active_count)
        legacy = np.array(case["exact_normal_traction_integral_m3"])
        row = dict(
            shape=shape.id,
            source_sha256=shape.provenance["source_sha256"],
            separation_m=case["separation_m"],
            target_sampled_depth_fraction=case["target_sampled_depth_fraction"],
            maximum_sampled_depth_fraction=maximum_depth,
            exact_normal_traction_integral_m3=force,
            exact_active_nodes=counts,
            legacy_relative_force_difference=float(
                np.linalg.norm(force - legacy) / np.linalg.norm(force)
            ),
            legacy_active_nodes=case["exact_active_nodes"],
        )
        rows.append(row)
        write_json(args.output / "progress.json", rows)
        print(row, flush=True)
    if sha256(args.protocol) != protocol_hash or any(
        sha256(root / name) != digest for name, digest in frozen.items()
    ):
        raise ValueError("Frozen reference inputs changed")
    write_json(
        args.output / "report.json",
        dict(
            protocol=policy,
            protocol_sha256=protocol_hash,
            source_hashes=frozen,
            rows=rows,
            runtime_versions={
                p: importlib.metadata.version(p)
                for p in ("numpy", "trimesh", "scipy", "rtree")
            },
            scope="Strict FP64 distance-minimum traction at the original 2048 source nodes and ten unchanged poses. Not surface-quadrature or material acceptance.",
        ),
    )


if __name__ == "__main__":
    main()
