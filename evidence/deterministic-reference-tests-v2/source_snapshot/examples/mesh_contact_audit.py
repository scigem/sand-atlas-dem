"""Separate SDF traction error from surface quadrature using source triangles."""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.benchmark import controlled_pairs
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.util import read_json, sha256, write_json


def traction(shape, points, areas, branch, query):
    force = np.zeros(3)
    counts = []
    for sign in (1, -1):
        phi, grad = query(points - sign * branch)
        active = phi < 0
        norms = np.linalg.norm(grad[active], axis=1)
        if np.any(norms < 1e-12):
            raise ValueError("Singular active normal")
        force += sign * np.sum(
            0.5
            * areas[active, None]
            * (-phi[active, None])
            * grad[active]
            / norms[:, None],
            axis=0,
        )
        counts.append(int(active.sum()))
    return force, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("protocol", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    policy = read_json(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    libraries = []
    for level in ("coarse", "fine"):
        path = Path(policy[f"{level}_library"])
        if sha256(path) != policy[f"{level}_sha256"]:
            raise ValueError("Library identity changed")
        libraries.append(ShapeLibrary.load(path))
    coarse, fine = libraries
    order = np.argsort([s.volume for s in fine.shapes])
    selected = order[np.array(policy["volume_order_indices"])]
    rows = []
    for index in selected:
        old, shape = coarse.shapes[index], fine.shapes[index]
        if old.id != shape.id or not np.array_equal(old.vertices, shape.vertices):
            raise ValueError("Source geometry changed")
        unit = (6 * shape.volume / np.pi) ** (1 / 3)
        mesh = trimesh.Trimesh(shape.vertices / unit, shape.faces, process=False)

        def exact(points, unit=unit, mesh=mesh):
            normalized = points / unit
            nearest, distance, _ = trimesh.proximity.closest_point(mesh, normalized)
            inside = mesh.contains(normalized)
            phi = np.where(inside, -distance, distance) * unit
            gradient = normalized - nearest
            gradient[inside] *= -1
            norm = np.linalg.norm(gradient, axis=1)
            gradient /= np.maximum(norm[:, None], 1e-30)
            return phi, gradient

        for depth in policy["placement_depth_fractions"]:
            # Fix a shared physical pose from the coarse field; never move the
            # refined case separately to manufacture matching forces.
            _, _, _, _, distances = controlled_pairs(
                ShapeLibrary([old]), 2, depth_fraction=depth
            )
            branch = np.array([distances[0], 0, 0])
            oracle, counts = traction(shape, shape.samples, shape.areas, branch, exact)
            a, _ = traction(old, shape.samples, shape.areas, branch, old.query)
            b, _ = traction(shape, shape.samples, shape.areas, branch, shape.query)
            norm = np.linalg.norm(oracle)
            if norm == 0:
                raise ValueError("No exact triangle contact in frozen pose")
            rows.append(
                {
                    "shape": shape.id,
                    "source_sha256": shape.provenance["source_sha256"],
                    "placement_depth_fraction": depth,
                    "separation_m": distances[0],
                    "exact_active_nodes": counts,
                    "exact_normal_traction_integral_m3": oracle.tolist(),
                    "coarse_relative_force_error": float(
                        np.linalg.norm(a - oracle) / norm
                    ),
                    "fine_relative_force_error": float(
                        np.linalg.norm(b - oracle) / norm
                    ),
                }
            )
    errors = [row["fine_relative_force_error"] for row in rows]
    passed = max(errors) <= policy["limits"]["maximum_relative_force_error"]
    write_json(
        args.output / "report.json",
        {
            "protocol": policy,
            "protocol_sha256": sha256(args.protocol),
            "script_sha256": sha256(__file__),
            "rows": rows,
            "maximum_fine_relative_force_error": max(errors),
            "passed": passed,
            "scope": "Same fine surface quadrature in both fields and triangle oracle; isolates field interpolation and normals, not quadrature or continuum convergence.",
        },
    )
    print(passed, max(errors), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
