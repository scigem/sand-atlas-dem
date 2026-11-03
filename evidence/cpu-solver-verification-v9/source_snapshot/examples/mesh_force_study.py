"""Actual GPU mesh-BVH forces against the unchanged frozen source-triangle oracle."""

import argparse
from pathlib import Path

import numpy as np

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("protocol", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    policy = read_json(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    library_path = Path(policy["library"])
    parent = Path(policy["reference_report"])
    if (
        sha256(library_path) != policy["library_sha256"]
        or sha256(parent) != policy["reference_report_sha256"]
    ):
        raise ValueError("Frozen source identity changed")
    shapes = {s.id: s for s in ShapeLibrary.load(library_path).shapes}
    metal = Metal()
    rows = []
    for case in read_json(parent)["rows"]:
        shape = shapes[case["shape"]]
        origin = np.full(3, 4 * shape.radius)
        positions = np.array([origin, origin + [case["separation_m"], 0, 0]])
        spec = Experiment(
            particles=2,
            timestep=policy["timestep_s"],
            contact_geometry="mesh",
            friction=0,
            max_overlap_fraction=policy["overlap_guard_fraction"],
            contacts_per_particle=4096,
            pairs_per_particle=4,
        )
        e = Engine(
            ShapeLibrary([shape]),
            spec,
            positions,
            np.full(3, 20 * shape.radius),
            [0, 0],
            [1.0, 1.0],
            metal=metal,
        )
        e.initialize()
        force = metal.read(e.buffers["force"]).reshape(-1, 4)[0, :3] * e.force_unit
        reference = (
            np.array(case["exact_normal_traction_integral_m3"]) * spec.normal_stiffness
        )
        error = float(np.linalg.norm(force - reference) / np.linalg.norm(reference))
        rows.append(
            {
                "shape": shape.id,
                "separation_m": case["separation_m"],
                "relative_force_error": error,
                "force_N": force.tolist(),
                "reference_N": reference.tolist(),
                "active_contacts": e.diagnostics()["active_surface_contacts"],
                "reference_active_contacts": sum(case["exact_active_nodes"]),
            }
        )
        print(rows[-1], flush=True)
    passed = max(row["relative_force_error"] for row in rows) <= policy[
        "maximum_relative_force_error"
    ] and all(
        row["active_contacts"] == row["reference_active_contacts"] for row in rows
    )
    write_json(
        args.output / "report.json",
        {
            "protocol": policy,
            "protocol_sha256": sha256(args.protocol),
            "script_sha256": sha256(__file__),
            "source_hashes": {
                p.name: sha256(p)
                for p in Path(__file__)
                .resolve()
                .parents[1]
                .joinpath("src/sand_atlas_dem")
                .glob("*.py")
            },
            "shader_hashes": {
                p.name: sha256(p)
                for p in Path(__file__)
                .resolve()
                .parents[1]
                .joinpath("src/sand_atlas_dem/shaders")
                .glob("*.slang")
            },
            "backend": metal.doctor(),
            "rows": rows,
            "passed": passed,
            "scope": "Exact-feature distance at fixed surface quadrature; no continuum quadrature, assembly or material acceptance.",
        },
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
