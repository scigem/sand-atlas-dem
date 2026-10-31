"""Fixed-area plane-contact oscillator versus a closed independent linear solution.

The cube has one face-centre quadrature node per face, each with its true face
area. Only the bottom face is active; its linear traction resultant is exact.
This is a declared analytical fixture, not a resolved mesh-assembly experiment.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.integrate import quad

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import read_json, sha256, write_json


def run(metal, policy, dt):
    h, depth, density = (
        policy["side_m"],
        policy["initial_depth_m"],
        policy["density_kg_m3"],
    )
    kn, cn, duration = (
        policy["stiffness_N_m3"],
        policy["viscosity_N_s_m3"],
        policy["duration_s"],
    )
    shape = from_mesh(
        trimesh.creation.box(extents=[h] * 3), unit="m", resolution=8, samples=16
    )
    shape.samples = np.concatenate([np.eye(3), -np.eye(3)]) * h / 2
    shape.areas = np.full(6, h * h)
    shape.provenance["quadrature"] = (
        "six exact face-centre areas; analytical linear-traction fixture"
    )
    shape.validate()
    z0 = 0.002
    wall = dict(
        normal=[0, 0, 1],
        offset=z0 - h / 2 + depth,
        velocity=[0, 0, 0],
        normal_stiffness=kn,
        tangential_stiffness=kn / 2,
        friction=0,
        normal_viscosity=cn,
    )
    spec = Experiment(
        periodic=False,
        particles=1,
        density=density,
        normal_stiffness=kn,
        tangential_stiffness=kn / 2,
        timestep=dt,
        planes=[wall],
        max_overlap_fraction=0.03,
    )
    engine = Engine(
        ShapeLibrary([shape]),
        spec,
        [[0.002, 0.002, z0]],
        [0.006] * 3,
        [0],
        [1],
        metal=metal,
    )
    engine.initialize()
    mass, stiffness, damping = density * h**3, kn * h * h, cn * h * h
    alpha, omega0 = damping / (2 * mass), np.sqrt(stiffness / mass)
    omega = np.sqrt(omega0**2 - alpha**2)

    def exact(t):
        decay = np.exp(-alpha * t)
        x = depth * decay * (np.cos(omega * t) + alpha / omega * np.sin(omega * t))
        velocity = -depth * decay * omega0**2 / omega * np.sin(omega * t)
        return x, velocity

    initial_energy = stiffness * depth**2 / 2
    steps = round(duration / dt)
    assert np.isclose(steps * dt, duration, rtol=1e-12, atol=0)
    max_error, min_traction, max_balance = 0.0, float("inf"), 0.0
    rows = []
    for step in range(1, steps + 1):
        engine.advance(1)
        t = step * dt
        x, v = exact(t)
        state, row = engine.state(), engine.diagnostics()
        position = (
            state["position"][0, 2].astype(float) + state["positionRemainder"][0, 2]
        ) * engine.length_unit
        velocity = (
            (state["velocity"][0, 2].astype(float) + state["velocityRemainder"][0, 2])
            * engine.length_unit
            / engine.time_unit
        )
        observed_depth = depth - (position - z0)
        error = max(
            abs(observed_depth - x) / depth, abs(velocity + v) / (depth * omega0)
        )
        max_error = max(max_error, error)
        min_traction = min(min_traction, kn * x + cn * v)
        assert row["active_surface_contacts"] == 1
        loss = row["normal_contact_dissipation"]
        balance = (
            abs(
                initial_energy
                - (row["kinetic_energy"] + row["elastic_energy_proxy"] + loss)
            )
            / initial_energy
        )
        max_balance = max(max_balance, balance)
        rows.append(
            dict(
                time_s=t,
                depth_m=observed_depth,
                velocity_m_s=velocity,
                exact_depth_m=x,
                exact_velocity_m_s=-v,
                loss_J=loss,
                relative_phase_error=error,
                relative_balance_error=balance,
            )
        )
    exact_loss = quad(lambda t: damping * exact(t)[1] ** 2, 0, duration, epsabs=1e-18)[
        0
    ]
    return dict(
        timestep_s=dt,
        steps=steps,
        maximum_relative_phase_error=max_error,
        maximum_relative_balance_error=max_balance,
        relative_loss_error=abs(loss / exact_loss - 1),
        exact_loss_J=exact_loss,
        minimum_exact_repulsive_traction=min_traction,
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = Path(__file__).resolve().parents[1], args.output
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        *sorted((root / "src/sand_atlas_dem").rglob("*.py")),
        *sorted((root / "src/sand_atlas_dem/shaders").glob("*.slang")),
        Path(__file__),
        args.protocol.resolve(),
    ]
    hashes = {}
    for path in paths:
        name = path.relative_to(root)
        hashes[str(name)] = sha256(path)
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(output / "prospective-freeze.json", dict(files=hashes))
    policy = read_json(args.protocol)
    write_json(output / "protocol.json", policy)
    metal = Metal()
    results = []
    for dt in policy["timesteps_s"]:
        result = run(metal, policy, dt)
        write_json(output / f"dt-{dt:.0e}.json", result)
        results.append({k: v for k, v in result.items() if k != "rows"})
    refinement = (
        results[0]["maximum_relative_phase_error"]
        / results[-1]["maximum_relative_phase_error"]
    )
    passed = bool(
        all(
            r["maximum_relative_phase_error"] < policy["maximum_phase_error"]
            and r["minimum_exact_repulsive_traction"] > 0
            for r in results
        )
        and results[-1]["maximum_relative_balance_error"]
        < policy["maximum_finest_balance_error"]
        and results[-1]["relative_loss_error"] < policy["maximum_finest_loss_error"]
        and refinement >= policy["minimum_refinement_factor"]
    )
    report = dict(
        identity=policy["identity"],
        backend="metal",
        cases=results,
        phase_refinement_factor=refinement,
        passed=passed,
        scientific_admission=False,
        scope=__doc__,
    )
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
