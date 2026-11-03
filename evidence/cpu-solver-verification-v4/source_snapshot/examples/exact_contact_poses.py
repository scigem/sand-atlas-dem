"""Freeze triangle-based contact poses; CPU source geometry, no dynamics."""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("protocol", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    policy = read_json(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    library = Path(policy["library"])
    parent = Path(policy["parent_report"])
    if (
        sha256(library) != policy["library_sha256"]
        or sha256(parent) != policy["parent_report_sha256"]
    ):
        raise ValueError("Source identity changed")
    shapes = {s.id: s for s in ShapeLibrary.load(library).shapes}
    rows = []
    for case in read_json(parent)["rows"]:
        sh = shapes[case["shape"]]
        unit = (6 * sh.volume / np.pi) ** (1 / 3)
        mesh = trimesh.Trimesh(sh.vertices / unit, sh.faces, process=False)

        def query(points, mesh=mesh, unit=unit):
            p = points / unit
            nearest, distance, _ = trimesh.proximity.closest_point(mesh, p)
            inside = mesh.contains(p)
            phi = np.where(inside, -distance, distance) * unit
            gradient = p - nearest
            gradient[inside] *= -1
            gradient /= np.maximum(np.linalg.norm(gradient, axis=1)[:, None], 1e-30)
            return phi, gradient

        def evaluate(separation, sh=sh, query=query):
            branch = np.array([separation, 0, 0])
            return query(np.vstack([sh.samples - branch, sh.samples + branch]))

        target = case["placement_depth_fraction"] * unit
        lower = case["separation_m"]
        upper = 2.1 * sh.radius
        original, _ = evaluate(lower)
        if -original.min() < target:
            raise ValueError("Frozen parent pose does not bracket the requested depth")
        outside, _ = evaluate(upper)
        if outside.min() < 0:
            raise ValueError("Bounding-sphere separation is not exterior")
        for _ in range(policy["bisection_iterations"]):
            middle = (lower + upper) / 2
            phi, _ = evaluate(middle)
            if -phi.min() >= target:
                lower = middle
            else:
                upper = middle
        separation = upper  # Choose the no-deeper side of the final bracket.
        phi, gradient = evaluate(separation)
        n = len(sh.samples)
        force = np.zeros(3)
        counts = []
        for i, sign in enumerate([1, -1]):
            p = phi[i * n : (i + 1) * n]
            g = gradient[i * n : (i + 1) * n]
            active = p < 0
            force += sign * np.sum(
                0.5 * sh.areas[active, None] * (-p[active, None]) * g[active], axis=0
            )
            counts.append(int(active.sum()))
        row = {
            "shape": sh.id,
            "source_sha256": sh.provenance["source_sha256"],
            "target_sampled_depth_fraction": target / unit,
            "actual_sampled_depth_fraction": float(-phi.min() / unit),
            "separation_m": separation,
            "parent_separation_m": case["separation_m"],
            "parent_actual_sampled_depth_fraction": float(-original.min() / unit),
            "exact_active_nodes": counts,
            "exact_normal_traction_integral_m3": force.tolist(),
        }
        rows.append(row)
        print(row, flush=True)
        write_json(args.output / "progress.json", {"rows": rows})
    write_json(
        args.output / "report.json",
        {
            "protocol": policy,
            "protocol_sha256": sha256(args.protocol),
            "script_sha256": sha256(__file__),
            "rows": rows,
            "scope": "Independent triangle-based placement and traction at fixed 2048-node surface quadrature. Maximum sampled depth is not an unsampled geometric intersection bound or continuum-quadrature verification.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
