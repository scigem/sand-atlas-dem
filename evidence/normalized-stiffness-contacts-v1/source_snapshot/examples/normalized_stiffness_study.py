"""Pressure-normalized stiffness: independent convex static roots and Metal dynamics."""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import trimesh
from contact_timestep_study import ResearchEngine
from scipy.optimize import brentq

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh, with_face_quadrature
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import quat_matrix, read_json, sha256, write_json


def fixture(name, diameter, level):
    if name == "faceted-sphere":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1)
    else:
        seed = trimesh.creation.icosphere(subdivisions=1, radius=1).vertices
        radii = (
            1
            + 0.12 * np.sin(3 * seed[:, 0] + 2 * seed[:, 1])
            + 0.08 * seed[:, 0] * seed[:, 2]
        )
        mesh = trimesh.convex.convex_hull(seed * radii[:, None] * [1, 0.8, 1.15])
    mesh.apply_scale(diameter / (6 * mesh.volume / np.pi) ** (1 / 3))
    shape = from_mesh(
        mesh,
        shape_id=name,
        unit="m",
        resolution=16,
        samples=16,
        provenance={
            "fixture": "Original procedural convex synthetic solid; mesh queries only"
        },
    )
    shape = with_face_quadrature(shape, "gauss3", level)
    q = (
        np.array([0, 0, 0, 1.0])
        if name == "faceted-sphere"
        else np.array([0, 0, math.sin(0.23), math.cos(0.23)])
    )
    transverse = (
        np.zeros(2) if name == "faceted-sphere" else diameter * np.array([0.12, 0.07])
    )
    return shape, np.array([[0, 0, 0, 1.0], q]), transverse


def oracle(shape, rotations, transverse, separation):
    """Exact convex signed interior distance from supporting face planes in FP64.

    A ball about an interior point first touches the nearest supporting face.
    Thus max signed half-space distance is the signed boundary distance inside.
    Exterior nodes contribute no traction; no exterior distance is needed.
    """
    mesh = trimesh.Trimesh(shape.vertices, shape.faces, process=False)
    normals = np.asarray(mesh.face_normals)
    offset = -np.einsum("ij,ij->i", normals, mesh.triangles[:, 0])
    low, high = shape.vertices.min(0), shape.vertices.max(0)
    centres = np.array([[0.0, 0, 0], [separation, *transverse]])
    rotation = [quat_matrix(q) for q in rotations]
    force = np.zeros(3)
    torque = np.zeros((2, 3))
    maxdepth = 0
    active = 0
    for source in [0, 1]:
        target = 1 - source
        arm = shape.samples @ rotation[source].T
        branch = centres[target] - centres[source]
        local = (arm - branch) @ rotation[target]
        indices = np.flatnonzero(np.all((local >= low) & (local <= high), axis=1))
        if not len(indices):
            continue
        signed = local[indices] @ normals.T + offset
        face = signed.argmax(axis=1)
        phi = signed[np.arange(len(indices)), face]
        keep = phi < 0
        indices = indices[keep]
        face = face[keep]
        depth = -phi[keep]
        if not len(indices):
            continue
        n = normals[face] @ rotation[target].T
        f = (
            0.5
            * (shape.areas[indices] * depth)[:, None]
            * n
            * (1 if source == 0 else -1)
        )
        force += f.sum(0)
        arms0 = arm[indices] + centres[source] - centres[0]
        arms1 = arm[indices] + centres[source] - centres[1]
        torque[0] += np.cross(arms0, f).sum(0)
        torque[1] += np.cross(arms1, -f).sum(0)
        maxdepth = max(maxdepth, float(depth.max()))
        active += len(indices)
    return dict(
        force_integral_m3=force,
        torque_integral_m4=torque,
        maximum_depth_m=maxdepth,
        active_contacts=active,
    )


def initial_engine(metal, shape, rotations, transverse, separation, kn, dt, policy):
    d = policy["reference_diameter_m"]
    centre = np.full(3, 3 * d)
    spec = Experiment(
        periodic=False,
        particles=2,
        density=policy["density_kg_m3"],
        contact_geometry="mesh",
        normal_stiffness=kn,
        tangential_stiffness=kn * policy["tangential_to_normal_stiffness"],
        friction=policy["friction"],
        timestep=dt,
        contacts_per_particle=8192,
        pairs_per_particle=4,
        bin_capacity=16,
        max_overlap_fraction=policy["diagnostic_overlap_guard_fraction"],
    )
    e = ResearchEngine(
        ShapeLibrary([shape]),
        spec,
        [centre, centre + [separation, *transverse]],
        np.full(3, 10 * d),
        [0, 0],
        [1, 1],
        metal=metal,
    )
    e.buffers["rotation"].copy_from_numpy(rotations.astype(np.float32))
    e.buffers["rotationRemainder"].copy_from_numpy(
        (rotations - rotations.astype(np.float32)).astype(np.float32)
    )
    e.initialize()
    return e


def observation(engine, origin):
    s = engine.state()
    row = engine.diagnostics()

    def value(name, unit):
        a = s[name].astype(float)
        if name + "Remainder" in s:
            a += s[name + "Remainder"]
        return a * unit

    return dict(
        step=engine.step,
        time_s=engine.step * engine.spec.timestep,
        position_m=(value("position", engine.length_unit)[:, :3] - origin).tolist(),
        velocity_m_s=value("velocity", engine.length_unit / engine.time_unit)[
            :, :3
        ].tolist(),
        rotation=value("rotation", 1).tolist(),
        force_N=(s["force"][:, :3] * engine.force_unit).tolist(),
        torque_N_m=(
            s["torque"][:, :3] * engine.force_unit * engine.length_unit
        ).tolist(),
        maximum_overlap_fraction=row["max_overlap_fraction"],
        active_contacts=row["active_surface_contacts"],
        kinetic_energy_J=row["kinetic_energy"],
        spring_energy_proxy_J=row["elastic_energy_proxy"],
        friction_loss_J=row["friction_dissipation"],
    )


def dynamic(metal, shape, rotations, transverse, separation, kn, dt, policy):
    e = initial_engine(metal, shape, rotations, transverse, separation, kn, dt, policy)
    d = policy["reference_diameter_m"]
    speed = math.sqrt(policy["reference_pressure_Pa"] / policy["density_kg_m3"])
    velocity = np.zeros((2, 4), np.float32)
    velocity[:, 1] = (
        np.array([-0.5, 0.5])
        * policy["initial_tangential_speed_scale"]
        * speed
        * e.time_unit
        / e.length_unit
    )
    e.buffers["velocity"].copy_from_numpy(velocity)
    # Re-evaluate forces at zero history increment with the prescribed initial velocity.
    e.initialize()
    interval = round(policy["observation_period_s"] / dt)
    total = round(policy["duration_s"] / dt)
    assert np.isclose(interval * dt, policy["observation_period_s"], rtol=1e-12, atol=0)
    assert np.isclose(total * dt, policy["duration_s"], rtol=1e-12, atol=0)
    origin = np.full(3, 3 * d)
    rows = [observation(e, origin)]
    failure = None
    try:
        for _ in range(total // interval):
            e.advance(interval)
            rows.append(observation(e, origin))
            if not all(
                np.isfinite(np.asarray(rows[-1][k])).all()
                for k in [
                    "position_m",
                    "velocity_m_s",
                    "rotation",
                    "force_N",
                    "torque_N_m",
                ]
            ):
                raise ValueError("Nonfinite observed trajectory")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    return dict(
        timestep_s=dt,
        production_preflight=e.production_preflight,
        completed=e.step == total and failure is None,
        failure=failure,
        rows=rows,
    )


def compare(candidate, reference, policy):
    if not candidate["completed"] or not reference["completed"]:
        return dict(accurate=False, reason="Incomplete candidate or reference")
    d = policy["reference_diameter_m"]
    p = policy["reference_pressure_Pa"]
    v = math.sqrt(p / policy["density_kg_m3"])
    scales = dict(position_m=d, velocity_m_s=v, force_N=p * d * d, torque_N_m=p * d**3)
    errors = {key: 0.0 for key in policy["accuracy_limits"]}
    assert len(candidate["rows"]) == len(reference["rows"])
    for a, b in zip(candidate["rows"], reference["rows"]):
        assert np.isclose(a["time_s"], b["time_s"], rtol=1e-12, atol=0)
        for field, scale in scales.items():
            key = field.split("_")[0]
            errors[key] = max(
                errors[key],
                float(
                    np.max(
                        np.linalg.norm(
                            (np.array(a[field]) - np.array(b[field])) / scale, axis=1
                        )
                    )
                ),
            )
        q = np.array(a["rotation"])
        r = np.array(b["rotation"])
        q /= np.linalg.norm(q, axis=1)[:, None]
        r /= np.linalg.norm(r, axis=1)[:, None]
        distance = np.minimum(
            np.linalg.norm(q - r, axis=1), np.linalg.norm(q + r, axis=1)
        )
        errors["rotation"] = max(
            errors["rotation"],
            float(np.max(4 * np.arcsin(np.clip(distance / 2, 0, 1)))),
        )
    return dict(
        accurate=all(
            errors[k] <= limit for k, limit in policy["accuracy_limits"].items()
        ),
        maximum_errors=errors,
        reference_is_self=candidate is reference,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        *sorted((root / "src/sand_atlas_dem").rglob("*.py")),
        *sorted((root / "src/sand_atlas_dem/shaders").glob("*.slang")),
        Path(__file__),
        root / "examples/contact_timestep_study.py",
        args.protocol.resolve(),
    ]
    hashes = {str(p.relative_to(root)): sha256(p) for p in paths}
    for p in paths:
        dest = out / "source_snapshot" / p.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    write_json(out / "prospective-freeze.json", dict(files=hashes))
    policy = read_json(args.protocol)
    d = policy["reference_diameter_m"]
    p = policy["reference_pressure_Pa"]
    target = p * d * d
    prepared = []
    static = []
    private = out / "private"
    private.mkdir()
    # All root finding is independent geometry analysis, never CPU dynamics.
    for name in policy["fixtures"]:
        shape, rotations, transverse = fixture(
            name, d, policy["quadrature_levels"][name]
        )
        fine = with_face_quadrature(
            shape, "gauss3", policy["quadrature_levels"][name] + 1
        )
        finest = with_face_quadrature(
            shape, "gauss3", policy["quadrature_levels"][name] + 2
        )
        asset = private / (name + ".h5")
        ShapeLibrary([shape]).save(asset)
        for kappa in policy["kappas"]:
            kn = kappa * p / d

            def load(separation):
                return (
                    -oracle(fine, rotations, transverse, separation)[
                        "force_integral_m3"
                    ][0]
                    * kn
                )

            sep = brentq(lambda x: load(x) - target, 0.25 * d, 2.5 * d, xtol=d * 1e-10)
            coarse = oracle(shape, rotations, transverse, sep)
            ref = oracle(fine, rotations, transverse, sep)
            challenge = oracle(finest, rotations, transverse, sep)
            norm = np.linalg.norm(ref["force_integral_m3"])
            eps = d * 1e-5
            tangent = (load(sep - eps) - load(sep + eps)) / (2 * eps)
            row = dict(
                fixture=name,
                kappa=kappa,
                normal_stiffness_N_m3=kn,
                reference_pressure_Pa=p,
                nominal_target_force_N=target,
                reference_diameter_m=d,
                separation_m=sep,
                rotations=rotations.tolist(),
                transverse_m=transverse.tolist(),
                library_sha256=sha256(asset),
                sample_count=len(shape.samples),
                coarse_reference=coarse,
                fine_reference=ref,
                finest_reference=challenge,
                coarse_quadrature_error=float(
                    np.linalg.norm(
                        coarse["force_integral_m3"] - ref["force_integral_m3"]
                    )
                    / norm
                ),
                reference_quadrature_change=float(
                    np.linalg.norm(
                        challenge["force_integral_m3"] - ref["force_integral_m3"]
                    )
                    / np.linalg.norm(challenge["force_integral_m3"])
                ),
                reference_overlap_fraction=ref["maximum_depth_m"] / d,
                production_overlap_comparison_passed=ref["maximum_depth_m"] / d
                <= policy["production_overlap_comparison_fraction"],
                normal_tangent_N_m=tangent,
                local_normal_period_s=2
                * np.pi
                / math.sqrt(2 * tangent / (policy["density_kg_m3"] * shape.volume)),
            )
            static.append(row)
            prepared.append((shape, rotations, transverse, sep, kn, row))
            write_json(
                out / "static-intake.json",
                dict(
                    cases=static,
                    scope="Independent convex geometry and fixed-quadrature static load roots; no dynamics",
                ),
            )
            print(
                "static",
                name,
                kappa,
                "overlap",
                row["reference_overlap_fraction"],
                "quadrature",
                row["coarse_quadrature_error"],
                flush=True,
            )
    metal = Metal()
    results = []
    for shape, rotations, transverse, sep, kn, row in prepared:
        fixture_name = row["fixture"]
        kappa = row["kappa"]
        runs = []
        for dt in policy["timesteps_s"]:
            try:
                result = dynamic(
                    metal, shape, rotations, transverse, sep, kn, dt, policy
                )
            except Exception as exc:
                result = dict(
                    timestep_s=dt,
                    completed=False,
                    failure=f"{type(exc).__name__}: {exc}",
                    rows=[],
                )
            file = out / f"{fixture_name}-k{kappa}-dt{dt:.0e}.json"
            write_json(file, result)
            runs.append(result)
            print(
                "dynamic",
                fixture_name,
                kappa,
                dt,
                result["completed"],
                result["failure"],
                flush=True,
            )
        reference = runs[0]
        comparison = [
            dict(timestep_s=r["timestep_s"], **compare(r, reference, policy))
            for r in runs
        ]
        reference_supported = bool(len(comparison) > 1 and comparison[1]["accurate"])
        initial = reference["rows"][0] if reference["rows"] else None
        if initial:
            expected = np.asarray(row["coarse_reference"]["force_integral_m3"]) * kn
            expected_torque = (
                np.asarray(row["coarse_reference"]["torque_integral_m4"]) * kn
            )
            observed = np.array(initial["force_N"])[0]
            row["gpu_oracle_force_error"] = float(
                np.linalg.norm(observed - expected) / np.linalg.norm(expected)
            )
            row["gpu_oracle_torque_error"] = float(
                np.max(
                    np.linalg.norm(
                        np.array(initial["torque_N_m"]) - expected_torque, axis=1
                    )
                )
                / (target * d)
            )
        item = dict(
            fixture=fixture_name,
            kappa=kappa,
            static=row,
            comparisons=comparison,
            reference_supported_by_2ns=reference_supported,
            largest_tested_accurate_timestep_s=max(
                (r["timestep_s"] for r in comparison[1:] if r["accurate"]), default=None
            ),
            reference_maximum_observed_overlap=max(
                (r["maximum_overlap_fraction"] for r in reference["rows"]), default=None
            ),
            note="Accuracy is agreement with a 1ns same-discretization trajectory, not an independent dynamic solution; 1ns versus2ns must also agree. Static quadrature checks and production-overlap comparison remain separate.",
        )
        results.append(item)
        write_json(
            out / "progress.json",
            dict(completed_pairs=len(results), expected_pairs=len(prepared)),
        )
    unchanged = all(sha256(root / path) == digest for path, digest in hashes.items())
    write_json(
        out / "report.json",
        dict(
            identity=policy["identity"],
            backend="metal",
            cases=results,
            sources_unchanged=unchanged,
            all_cases_recorded=len(results)
            == len(policy["fixtures"]) * len(policy["kappas"]),
            scientific_admission=False,
            scope=policy["scope"],
        ),
    )
    if not unchanged:
        raise SystemExit("Source integrity failed")


if __name__ == "__main__":
    main()
