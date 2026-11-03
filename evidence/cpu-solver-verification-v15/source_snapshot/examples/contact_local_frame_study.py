"""Independent SI plane-traction checks under large whole-fixture translations."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        *sorted((root / "src/sand_atlas_dem").rglob("*.py")),
        *sorted((root / "src/sand_atlas_dem/shaders").glob("*.slang")),
        Path(__file__),
        args.protocol.resolve(),
    ]
    hashes = {str(p.relative_to(root)): sha256(p) for p in paths}
    for p in paths:
        dest = args.output / "source_snapshot" / p.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    write_json(args.output / "prospective-freeze.json", dict(files=hashes))
    policy = read_json(args.protocol)
    h = policy["side_m"]
    depth = h * policy["depth_fraction"]
    kn = 1e12
    shape = from_mesh(
        trimesh.creation.box(extents=[h] * 3), unit="m", resolution=8, samples=16
    )
    shape.samples = np.concatenate([np.eye(3), -np.eye(3)]) * h / 2
    shape.areas = np.full(6, h * h)
    shape.provenance["quadrature"] = (
        "Exact face-centre areas for analytical traction verification"
    )
    shape.validate()
    metal = Metal()
    rows = []
    for normal in policy["normals"]:
        n = np.array(normal)
        for c in policy["centres_m"]:
            centre = np.full(3, c)
            offset = float(centre @ n + np.min(shape.samples @ n) + depth)
            wall = dict(
                normal=normal,
                offset=offset,
                velocity=[0, 0, 0],
                normal_stiffness=kn,
                tangential_stiffness=kn / 2,
                friction=0,
            )
            spec = Experiment(
                periodic=False,
                particles=1,
                timestep=2e-9,
                planes=[wall],
                contacts_per_particle=64,
                pairs_per_particle=8,
                bin_capacity=16,
                max_overlap_fraction=0.1,
            )
            engine = Engine(
                ShapeLibrary([shape]),
                spec,
                [centre],
                np.full(3, max(0.01, 2 * c)),
                [0],
                [1],
                metal=metal,
            )
            engine.initialize()
            state = engine.state()
            # SI reference evaluates local surface projection separately in FP64.
            penetration = np.maximum(0, offset - centre @ n - shape.samples @ n)
            forces = kn * (shape.areas * penetration)[:, None] * n
            expected_force = forces.sum(axis=0)
            expected_torque = np.cross(shape.samples, forces).sum(axis=0)
            force = state["force"][0, :3].astype(float) * engine.force_unit
            torque = (
                state["torque"][0, :3].astype(float)
                * engine.force_unit
                * engine.length_unit
            )
            force_scale = np.linalg.norm(expected_force)
            torque_scale = force_scale * h
            position = (
                state["position"][0, :3].astype(float)
                + state["positionRemainder"][0, :3]
            ) * engine.length_unit
            row = dict(
                normal=normal,
                centre_m=c,
                force_N=force.tolist(),
                reference_force_N=expected_force.tolist(),
                torque_N_m=torque.tolist(),
                reference_torque_N_m=expected_torque.tolist(),
                relative_force_error=float(
                    np.linalg.norm(force - expected_force) / force_scale
                ),
                relative_torque_error=float(
                    np.linalg.norm(torque - expected_torque) / torque_scale
                ),
                normalized_initial_position_error=float(
                    np.linalg.norm(position - centre) / h
                ),
                active_contacts=len(state["meta"]),
                expected_contacts=int(np.count_nonzero(penetration)),
            )
            row["passed"] = bool(
                row["relative_force_error"] <= policy["maximum_relative_force_error"]
                and row["relative_torque_error"]
                <= policy["maximum_relative_torque_error"]
                and row["normalized_initial_position_error"]
                <= policy["maximum_normalized_initial_position_error"]
                and row["active_contacts"] == row["expected_contacts"]
            )
            rows.append(row)
            print(
                c,
                normal,
                row["relative_force_error"],
                row["relative_torque_error"],
                row["passed"],
                flush=True,
            )
    unchanged = all(sha256(root / p) == digest for p, digest in hashes.items())
    write_json(
        args.output / "report.json",
        dict(
            identity=policy["identity"],
            backend="metal",
            cases=rows,
            sources_unchanged=unchanged,
            passed=unchanged and all(r["passed"] for r in rows),
            scientific_admission=False,
            scope=policy["scope"],
        ),
    )
    if not unchanged or not all(r["passed"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
