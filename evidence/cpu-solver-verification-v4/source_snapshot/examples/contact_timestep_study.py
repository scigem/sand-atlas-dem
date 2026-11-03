"""Actual GPU timestep sweep against independent unilateral and bilateral oscillators."""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import read_json, sha256, write_json


class ResearchEngine(Engine):
    def _preflight_dt(self):
        # Explicitly confined to this research harness. Production Engine is unchanged.
        try:
            super()._preflight_dt()
            self.production_preflight = "accepted"
        except ValueError as exc:
            self.production_preflight = str(exc)


def exact(mode, amplitude, omega, t):
    if mode == "confined" or omega * t <= math.pi / 2:
        return amplitude * math.cos(omega * t), -amplitude * omega * math.sin(omega * t)
    return -amplitude * omega * (t - math.pi / (2 * omega)), -amplitude * omega


def run(metal, policy, shape, mode, dt):
    h = (
        float(shape.vertices[:, 0].ptp())
        if hasattr(shape.vertices[:, 0], "ptp")
        else float(np.ptp(shape.vertices[:, 0]))
    )
    rho, kn = policy["density_kg_m3"], policy["normal_stiffness_N_m3"]
    a, d = h * policy["amplitude_fraction"], h * policy["preload_depth_fraction"]
    mass, k = rho * h**3, kn * h * h
    omega = math.sqrt((2 if mode == "confined" else 1) * k / mass)
    centre = (
        h * policy["centre_in_side_lengths"]
        if "centre_in_side_lengths" in policy
        else 0.005
    )
    wall = dict(
        normal=[0, 0, 1],
        offset=centre - h / 2 + (d if mode == "confined" else 0),
        velocity=[0, 0, 0],
        normal_stiffness=kn,
        tangential_stiffness=policy["tangential_stiffness_N_m3"],
        friction=0,
        normal_viscosity=0,
    )
    planes = [wall]
    if mode == "confined":
        planes.append(dict(wall, normal=[0, 0, -1], offset=-(centre + h / 2 - d)))
    spec = Experiment(
        periodic=False,
        particles=1,
        density=rho,
        normal_stiffness=kn,
        tangential_stiffness=policy["tangential_stiffness_N_m3"],
        friction=0,
        timestep=dt,
        planes=planes,
        contacts_per_particle=64,
        pairs_per_particle=8,
        bin_capacity=16,
        max_overlap_fraction=0.1,
    )
    e = ResearchEngine(
        ShapeLibrary([shape]),
        spec,
        [[centre, centre, centre + (a if mode == "confined" else -a)]],
        [0.01] * 3,
        [0],
        [1],
        metal=metal,
    )
    e.initialize()
    # Independently reproduce the existing conservative estimate, including rotation.
    stiff = (
        (kn + policy["tangential_stiffness_N_m3"])
        * (1 + len(planes))
        * shape.areas.sum()
    )
    bound = 0.1 * math.sqrt(
        min(
            mass / stiff,
            np.linalg.eigvalsh(shape.inertia * rho).min() / (stiff * shape.radius**2),
        )
    )
    total = max(1, math.ceil(policy["duration_omega_t"] / omega / dt))
    dynamic_energy = 0.5 * mass * (a * omega) ** 2
    rows = []
    failure = None
    domain_exit = False
    try:
        for step in range(0, total + 1):
            if (
                step
                and step % policy["maximum_steps_per_observation"]
                and step != total
            ):
                continue
            if step:
                e.advance(step - e.step)
            state = e.state()
            z = (
                float(state["position"][0, 2]) + float(state["positionRemainder"][0, 2])
            ) * e.length_unit
            v = (
                (
                    float(state["velocity"][0, 2])
                    + float(state["velocityRemainder"][0, 2])
                )
                * e.length_unit
                / e.time_unit
            )
            x = (z - centre) if mode == "confined" else (centre - z)
            velocity = v if mode == "confined" else -v
            t = step * dt
            refx, refv = exact(mode, a, omega, t)
            # Compute potential independently from physical plane penetration.
            if mode == "confined":
                potential = (
                    0.5 * k * (max(d - x, 0) ** 2 + max(d + x, 0) ** 2) - k * d * d
                )
                domain_exit = domain_exit or abs(x) >= d
            else:
                potential = 0.5 * k * max(x, 0) ** 2
            energy = 0.5 * mass * velocity**2 + potential
            row = dict(
                step=step,
                time_s=t,
                displacement_m=x,
                velocity_m_s=velocity,
                exact_displacement_m=refx,
                exact_velocity_m_s=refv,
                normalized_path_error=max(
                    abs(x - refx) / a, abs(velocity - refv) / (a * omega)
                ),
                relative_dynamic_energy_error=abs(energy / dynamic_energy - 1),
                dynamic_energy_multiple=energy / dynamic_energy,
                active_contacts=len(state["meta"]),
            )
            rows.append(row)
            if not all(
                np.isfinite(row[key])
                for key in ("displacement_m", "velocity_m_s", "dynamic_energy_multiple")
            ):
                raise ValueError("Nonfinite observed trajectory")
            if (
                row["dynamic_energy_multiple"]
                > policy["bounded_dynamic_energy_multiple"]
            ):
                raise ValueError("Prospective dynamic energy divergence stop")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    maxpath = max((r["normalized_path_error"] for r in rows), default=math.inf)
    maxenergy = max(
        (r["relative_dynamic_energy_error"] for r in rows), default=math.inf
    )
    complete = failure is None and e.step == total
    return dict(
        mode=mode,
        side_m=h,
        timestep_s=dt,
        omega_rad_s=omega,
        omega_dt=omega * dt,
        linear_verlet_stability_threshold_s=2 / omega,
        production_bound_s=bound,
        production_preflight=e.production_preflight,
        requested_steps=total,
        completed_step=e.step,
        failure=failure,
        analytical_domain_exit=domain_exit,
        completed=complete,
        bounded_observed=complete
        and not domain_exit
        and maxenergy < policy["bounded_dynamic_energy_multiple"] - 1,
        accurate=complete
        and not domain_exit
        and maxpath <= policy["maximum_normalized_path_error"]
        and maxenergy <= policy["maximum_relative_dynamic_energy_error"],
        maximum_normalized_path_error=maxpath,
        maximum_relative_dynamic_energy_error=maxenergy,
        rows=rows,
    )


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
    metal = Metal()
    results = []
    for h in policy["sizes_m"]:
        shape = from_mesh(
            trimesh.creation.box(extents=[h] * 3), unit="m", resolution=8, samples=16
        )
        shape.samples = np.concatenate([np.eye(3), -np.eye(3)]) * h / 2
        shape.areas = np.full(6, h * h)
        shape.provenance["quadrature"] = (
            "Six exact face-centre areas; analytical contact fixture"
        )
        shape.validate()
        for mode in policy["modes"]:
            omega = math.sqrt(
                (2 if mode == "confined" else 1)
                * policy["normal_stiffness_N_m3"]
                / (policy["density_kg_m3"] * h)
            )
            timesteps = sorted(
                set(
                    policy["physical_timesteps_s"]
                    + [q / omega for q in policy["dimensionless_timesteps"]]
                )
            )
            for index, dt in enumerate(timesteps):
                result = run(metal, policy, shape, mode, dt)
                write_json(args.output / f"{mode}-{h:.0e}-{index:02}.json", result)
                results.append({k: v for k, v in result.items() if k != "rows"})
                print(
                    mode,
                    h,
                    dt,
                    "accurate",
                    result["accurate"],
                    "error",
                    result["maximum_normalized_path_error"],
                    "failure",
                    result["failure"],
                    flush=True,
                )
    unchanged = all(sha256(root / p) == digest for p, digest in hashes.items())
    write_json(
        args.output / "report.json",
        dict(
            identity=policy["identity"],
            backend="metal",
            sources_unchanged=unchanged,
            cases=results,
            scientific_admission=False,
            scope=policy["scope"],
            all_cases_recorded=len(results)
            == len(policy["sizes_m"]) * len(policy["modes"]) * 16,
        ),
    )
    if not unchanged:
        raise SystemExit("Source integrity failed")


if __name__ == "__main__":
    main()
