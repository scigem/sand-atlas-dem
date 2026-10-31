"""Frozen Metal plane-operator verification against analytical surface integrals."""

import argparse
import importlib.util
import shutil
from pathlib import Path

import numpy as np

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import sphere
from sand_atlas_dem.util import read_json, sha256, write_json

path = Path(__file__).with_name("plane_reference.py")
spec = importlib.util.spec_from_file_location("plane_reference", path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def evaluate(metal, points, areas, velocity, old, *, normal, offset, kn, kt, mu, dt):
    count = len(points)
    buffers = dict(
        pointArea=metal.buffer(np.column_stack([points, areas]).astype(np.float32)),
        relativeVelocity=metal.buffer(
            np.column_stack([velocity, np.zeros(count)]).astype(np.float32)
        ),
        oldSlipActive=metal.buffer(np.asarray(old, np.float32)),
        forceEnergy=metal.buffer(count=count, stride=16),
        slipLoss=metal.buffer(count=count, stride=16),
        torqueActive=metal.buffer(count=count, stride=16),
    )
    metal.kernel("planeProbe", "plane.slang").dispatch(
        thread_count=[count, 1, 1],
        N=count,
        normal=normal,
        offset=offset,
        kn=kn,
        kt=kt,
        mu=mu,
        dt=dt,
        **buffers,
    )
    metal.device.wait()
    return {
        name: metal.read(buffers[name]).reshape(-1, 4).astype(float)
        for name in ("forceEnergy", "slipLoss", "torqueActive")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = read_json(args.protocol)
    protocol_hash = sha256(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        path,
        root / "src/sand_atlas_dem/shaders/plane.slang",
        root / "src/sand_atlas_dem/backend.py",
        root / "src/sand_atlas_dem/geometry.py",
        root / "src/sand_atlas_dem/util.py",
    ]
    sources = {str(p.relative_to(root)): sha256(p) for p in files}
    for file in files:
        target = args.output / "source_snapshot" / file.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    shutil.copy2(args.protocol, args.output / "protocol.json")
    write_json(
        args.output / "prospective-freeze.json",
        dict(protocol_sha256=protocol_hash, source_hashes=sources),
    )
    metal = Metal()
    point = np.array([[0.0, 0.0, -0.8]])
    area = np.array([0.2])
    initial = evaluate(
        metal,
        point,
        area,
        np.array([[0.0025, 0.0, 0.0]]),
        np.zeros((1, 4)),
        normal=[0.0, 0.0, 1.0],
        offset=-0.75,
        kn=1.0,
        kt=0.5,
        mu=0.3,
        dt=1.0,
    )
    old = np.column_stack([initial["slipLoss"][:, :3], np.ones(1)])
    continued = evaluate(
        metal,
        point,
        area,
        np.array([[0.0025, 0.0, 0.0]]),
        old,
        normal=[0.0, 0.0, 1.0],
        offset=-0.75,
        kn=1.0,
        kt=0.5,
        mu=0.3,
        dt=1.0,
    )
    separated = evaluate(
        metal,
        -point,
        area,
        np.zeros((1, 3)),
        old,
        normal=[0.0, 0.0, 1.0],
        offset=-0.75,
        kn=1.0,
        kt=0.5,
        mu=0.3,
        dt=1.0,
    )
    inactive = np.column_stack([continued["slipLoss"][:, :3], np.zeros(1)])
    returned = evaluate(
        metal,
        point,
        area,
        np.zeros((1, 3)),
        inactive,
        normal=[0.0, 0.0, 1.0],
        offset=-0.75,
        kn=1.0,
        kt=0.5,
        mu=0.3,
        dt=1.0,
    )
    clipped = evaluate(
        metal,
        point,
        area,
        np.array([[1.0, 0.0, 0.0]]),
        np.zeros((1, 4)),
        normal=[0.0, 0.0, 1.0],
        offset=-0.75,
        kn=1.0,
        kt=0.5,
        mu=0.3,
        dt=1.0,
    )
    expected = np.array([-0.003, 0.0, 0.01])
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = np.eye(3) + np.sin(0.7) * cross + (1 - np.cos(0.7)) * (cross @ cross)
    rotated = evaluate(
        metal,
        point @ rotation.T,
        area,
        np.array([[1.0, 0.0, 0.0]]) @ rotation.T,
        np.zeros((1, 4)),
        normal=(rotation @ np.array([0.0, 0.0, 1.0])).tolist(),
        offset=-0.75,
        kn=1.0,
        kt=0.5,
        mu=0.3,
        dt=1.0,
    )
    nodal_errors = dict(
        initial_force=float(
            np.linalg.norm(initial["forceEnergy"][0, :3] - [-0.00025, 0.0, 0.01]) / 0.01
        ),
        continued_force=float(
            np.linalg.norm(continued["forceEnergy"][0, :3] - [-0.0005, 0.0, 0.01])
            / 0.01
        ),
        returned_force=float(
            np.linalg.norm(returned["forceEnergy"][0, :3] - [0.0, 0.0, 0.01]) / 0.01
        ),
        clipped_force=float(
            np.linalg.norm(clipped["forceEnergy"][0, :3] - expected) / 0.01
        ),
        clipped_loss=float(abs(clipped["slipLoss"][0, 3] - 0.00291) / 0.00291),
        rotated_force=float(
            np.linalg.norm(rotated["forceEnergy"][0, :3] - rotation @ expected) / 0.01
        ),
        rotated_torque=float(
            np.linalg.norm(
                rotated["torqueActive"][0, :3] - rotation @ np.array([0.0, 0.0024, 0.0])
            )
            / 0.01
        ),
    )
    nodal_passed = all(
        v <= policy["maximum_nodal_relative_error"] for v in nodal_errors.values()
    ) and all(np.all(v == 0) for v in separated.values())
    rows = []
    for count in policy["sample_counts"]:
        shape = sphere(radius=1.0, resolution=8, samples=count)
        points, areas = shape.samples, shape.areas
        for depth in policy["depths"]:
            slip = (
                policy["transition_fraction"]
                * policy["mu"]
                * policy["kn"]
                * depth
                / policy["kt"]
            )
            expected = reference.sphere_plane(
                1.0, depth, policy["kn"], policy["kt"], policy["mu"], slip
            )
            for frictional in (False, True):
                velocity = np.zeros_like(points)
                velocity[:, 0] = slip if frictional else 0.0
                result = evaluate(
                    metal,
                    points,
                    areas,
                    velocity,
                    np.zeros((count, 4)),
                    normal=[0.0, 0.0, 1.0],
                    offset=-1.0 + depth,
                    kn=policy["kn"],
                    kt=policy["kt"],
                    mu=policy["mu"],
                    dt=1.0,
                )
                force = result["forceEnergy"][:, :3].sum(axis=0)
                torque = result["torqueActive"][:, :3].sum(axis=0)
                target_force = (
                    expected["force"]
                    if frictional
                    else np.array([0.0, 0.0, expected["force"][2]])
                )
                target_torque = expected["torque"] if frictional else np.zeros(3)
                # A single normal-force moment scale also covers the zero target torque.
                force_scale = expected["force"][2]
                row = dict(
                    samples=count,
                    depth=depth,
                    frictional=frictional,
                    force=force,
                    torque=torque,
                    relative_force_error=float(
                        np.linalg.norm(force - target_force) / force_scale
                    ),
                    relative_torque_error=float(
                        np.linalg.norm(torque - target_torque) / force_scale
                    ),
                    active_nodes=int(result["torqueActive"][:, 3].sum()),
                    finite=all(np.isfinite(v).all() for v in result.values()),
                )
                if not frictional:
                    row["relative_normal_energy_error"] = float(
                        abs(
                            result["forceEnergy"][:, 3].sum()
                            / expected["normal_energy"]
                            - 1
                        )
                    )
                rows.append(row)
                write_json(args.output / "progress.json", dict(rows=rows))
    fine = [row for row in rows if row["samples"] == policy["sample_counts"][-1]]
    passed = (
        nodal_passed
        and all(row["finite"] for row in rows)
        and all(
            max(
                row["relative_force_error"],
                row["relative_torque_error"],
                row.get("relative_normal_energy_error", 0.0),
            )
            <= policy["maximum_fine_relative_error"]
            for row in fine
        )
    )
    if sha256(args.protocol) != protocol_hash or any(
        sha256(root / p) != h for p, h in sources.items()
    ):
        raise ValueError("Frozen operator study changed")
    write_json(
        args.output / "report.json",
        dict(
            protocol=policy,
            protocol_sha256=protocol_hash,
            source_hashes=sources,
            nodal_errors=nodal_errors,
            nodal_passed=nodal_passed,
            rows=rows,
            passed=passed,
            backend=metal.doctor(),
            scope=policy["scope"],
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
