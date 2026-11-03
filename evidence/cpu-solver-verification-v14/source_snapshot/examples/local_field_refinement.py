"""Evaluate only queried grid corners; estimate resolution without baking full cubes."""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from mesh_contact_audit import traction

from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.util import read_json, sha256, write_json


def local_query(shape, mesh, unit, resolution):
    dx = np.ptp(shape.vertices, axis=0).max() / (resolution - 1)
    origin = shape.vertices.min(axis=0) - 3 * dx
    maximum = shape.vertices.max(axis=0)

    def query(points):
        points = np.atleast_2d(points)
        valid = np.all(
            (points >= shape.vertices.min(axis=0) - dx) & (points <= maximum + dx),
            axis=1,
        )
        value = np.full(len(points), np.inf)
        gradient = np.zeros((len(points), 3))
        if not valid.any():
            return value, gradient
        coordinates = (points[valid] - origin) / dx
        base = np.floor(coordinates).astype(int)
        t = coordinates - base
        bits = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        indices = base[:, None, :] + bits[None, :, :]
        unique, inverse = np.unique(indices.reshape(-1, 3), axis=0, return_inverse=True)
        corners = (origin + dx * unique) / unit
        # Match baked storage: exact source distance rounded to FP32 before interpolation.
        field = (
            (-trimesh.proximity.signed_distance(mesh, corners) * unit)
            .astype(np.float32)
            .astype(float)
        )
        field = field[inverse].reshape(len(base), 8)
        v = np.zeros(len(base))
        g = np.zeros((len(base), 3))
        for node, bit in enumerate(bits):
            weights = np.where(bit, t, 1 - t)
            v += field[:, node] * weights.prod(axis=1)
            for axis in range(3):
                others = [j for j in range(3) if j != axis]
                g[:, axis] += (
                    field[:, node]
                    * (2 * bit[axis] - 1)
                    * weights[:, others].prod(axis=1)
                    / dx
                )
        value[valid] = v
        gradient[valid] = g
        return value, gradient

    return query


def main():
    p = argparse.ArgumentParser()
    p.add_argument("protocol", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    policy = read_json(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    path = Path(policy["library"])
    if sha256(path) != policy["library_sha256"]:
        raise ValueError("Library changed")
    parent = Path(policy["parent_report"])
    if sha256(parent) != policy["parent_report_sha256"]:
        raise ValueError("Frozen contact poses changed")
    shapes = {s.id: s for s in ShapeLibrary.load(path).shapes}
    rows = []
    for case in read_json(parent)["rows"]:
        shape = shapes[case["shape"]]
        unit = (6 * shape.volume / np.pi) ** (1 / 3)
        mesh = trimesh.Trimesh(shape.vertices / unit, shape.faces, process=False)
        branch = np.array([case["separation_m"], 0, 0])
        reference = np.array(case["exact_normal_traction_integral_m3"])
        errors = {}
        for resolution in policy["resolutions"]:
            force, _ = traction(
                shape,
                shape.samples,
                shape.areas,
                branch,
                local_query(shape, mesh, unit, resolution),
            )
            errors[str(resolution)] = float(
                np.linalg.norm(force - reference) / np.linalg.norm(reference)
            )
        rows.append(
            {"shape": shape.id, "separation_m": case["separation_m"], "errors": errors}
        )
        print(rows[-1], flush=True)
        write_json(args.output / "progress.json", {"rows": rows})
    maxima = {
        str(r): max(row["errors"][str(r)] for row in rows)
        for r in policy["resolutions"]
    }
    write_json(
        args.output / "report.json",
        {
            "protocol": policy,
            "protocol_sha256": sha256(args.protocol),
            "script_sha256": sha256(__file__),
            "traction_script_sha256": sha256(
                Path(__file__).with_name("mesh_contact_audit.py")
            ),
            "rows": rows,
            "worst_relative_force_error": maxima,
            "scope": "Source geometry unchanged; only queried corners evaluated. Not a full baked asset, GPU execution, surface-quadrature convergence or material acceptance.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
