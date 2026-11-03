"""Frozen dense mechanical annealing study on the actual Metal mesh solver."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np

from sand_atlas_dem.annealing import agitation, cooling_energy, select_timestep, construction_quadrature
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.gpu import Engine, RejectedStep
from sand_atlas_dem.simulation import initial_assembly
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import quat_matrix, sha256, write_json


def snapshot(engine, path, phase, elapsed_time):
    """Construction state has explicit time; it is not a standard lab checkpoint."""
    with h5py.File(path, "w") as f:
        f.attrs.update(format="sand-dem-annealing-state/1", phase=phase,
                       step=engine.step, elapsed_pseudo_time_s=elapsed_time,
                       experiment=json.dumps(engine.spec.to_dict()),
                       shape_sha256=sha256(engine.spec.shape_library),
                       length_unit=engine.length_unit, time_unit=engine.time_unit)
        f.create_dataset("shape_ids", data=engine.ids)
        f.create_dataset("scales", data=engine.scales)
        for name, value in engine.state().items():
            if name == "contacts":
                f.attrs[name] = value
            else:
                f.create_dataset(name, data=value, compression="gzip")


def make_engine(library, spec, positions, cell, ids, scales, rotations, metal=None):
    attempts = []
    for retry in range(8):
        engine = Engine(library, spec, positions, cell, ids, scales, metal=metal)
        engine.buffers["rotation"].copy_from_numpy(rotations.astype(np.float32))
        try:
            engine.initialize()
            engine.capacity_retries = attempts
            return engine
        except RejectedStep as exc:
            attempts.append({"initialization": True, "flags": exc.flags,
                             "contacts_per_particle": spec.contacts_per_particle,
                             "pairs_per_particle": spec.pairs_per_particle,
                             "bin_capacity": spec.bin_capacity})
            print("Initial capacity rejection", attempts[-1], flush=True)
            if exc.flags & ~(8 | 16 | 32 | 64) or retry == 7:
                raise
            changes = {}
            if exc.flags & 8:
                changes["bin_capacity"] = spec.bin_capacity * 2
            if exc.flags & 16:
                changes["pairs_per_particle"] = spec.pairs_per_particle * 2
            if exc.flags & (32 | 64):
                changes["contacts_per_particle"] = spec.contacts_per_particle * 2
            metal = engine.m
            spec = replace(spec, **changes)
            del engine
    raise RuntimeError("Initialization capacity retries exhausted")


def worker(out):
    p = json.loads((out / "protocol.json").read_text())
    freeze = json.loads((out / "prospective-freeze.json").read_text())
    assert all(sha256(path) == value for path, value in freeze["inputs"].items())
    assert all(sha256(out / "source_snapshot" / name) == value
               for name, value in freeze["sources"].items())
    with h5py.File(p["seed_checkpoint"]) as f:
        original = Experiment(**json.loads(f.attrs["experiment"]))
        old = {name: f[name][:] for name in ["position", "positionRemainder", "cell", "rotation", "rotationRemainder"]}
    library = ShapeLibrary.load(original.shape_library)
    _, _, ids, scales = initial_assembly(library, original)
    length_unit = float(np.median([(6 * s.volume / np.pi)**(1/3) for s in library.shapes]))
    positions = (old["position"][:, :3].astype(float) + old["positionRemainder"][:, :3]) * length_unit
    cell = old["cell"][0, :3].astype(float) * length_unit
    volume = sum(library.shapes[int(i)].volume * s**3 for i, s in zip(ids, scales))
    factor = ((volume / np.prod(cell)) / p["construction_solid_fraction"])**(1/3)
    positions *= factor
    cell *= factor
    rotations = old["rotation"].astype(float) + old["rotationRemainder"]
    rotations /= np.linalg.norm(rotations, axis=1)[:, None]
    soft = replace(original, normal_stiffness=original.normal_stiffness*p["soft_stiffness_ratio"],
                   tangential_stiffness=original.tangential_stiffness*p["soft_stiffness_ratio"],
                   contact_normal=p.get("construction_contact_normal", "target-gradient"),
                   friction=0., normal_viscosity=0., timestep_policy="active-contact",
                   timestep=p["minimum_timestep_s"], max_overlap_fraction=p["construction_overlap_guard"])
    construction_library = library
    if p.get("construction_samples_per_shape"):
        coarse = []
        mapping = {}
        for index, shape in enumerate(library.shapes):
            candidate, selected = construction_quadrature(shape, p["construction_samples_per_shape"], p["seed"]+index)
            coarse.append(candidate)
            mapping[shape.id] = selected.tolist()
        construction_library = ShapeLibrary(coarse)
        path = out / "construction-shapes.h5"
        construction_library.save(path)
        soft = replace(soft, shape_library=str(path))
        write_json(out / "construction-quadrature.json", {"shape_library_sha256": sha256(path),
            "method": "Explicit artificial construction quadrature only; physical stage uses original library",
            "source_sample_indices": mapping})
    engine = None
    phase = "initialization"
    started = time.monotonic()
    physical_time = 0.
    total_steps = 0
    rejections = []
    rows = []
    thermostat_work = 0.
    converged = 0
    failure = None
    dt_values = []
    try:
        engine = make_engine(construction_library, soft, positions, cell, ids, scales, rotations)
        write_json(out / "initial-capacity-retries.json", engine.capacity_retries)
        initial = engine.diagnostics()
        write_json(out / "initial.json", initial)
        snapshot(engine, out / "construction-initial.h5", "construction", 0.)
        rng = np.random.default_rng(p["seed"])
        phase = "construction"
        with (out / "telemetry.jsonl").open("w") as log:
            for phase, budget in [("construction", p["construction_steps"]), ("physical-equilibration", p["equilibration_steps"])]:
                if phase == "physical-equilibration":
                    row = engine.diagnostics()
                    snapshot(engine, out / "construction-final.h5", "construction", physical_time)
                    if row["max_overlap_fraction"] > original.max_overlap_fraction:
                        raise RuntimeError("Construction did not reach unchanged physical overlap guard; no physical handoff")
                    state = engine.state()
                    positions = (state["position"][:, :3].astype(float) + state["positionRemainder"][:, :3]) * engine.length_unit
                    cell = state["cell"][0, :3].astype(float) * engine.length_unit
                    q = state["rotation"].astype(float) + state["rotationRemainder"]
                    q /= np.linalg.norm(q, axis=1)[:, None]
                    # Explicit event: artificial kinetic energy and friction history
                    # are discarded, physical contacts rebuilt at intended settings.
                    previous = engine
                    engine = make_engine(library, replace(original, timestep_policy="active-contact", timestep=p["minimum_timestep_s"]), positions, cell, ids, scales, q, previous.m)
                    del previous
                    write_json(out / "handoff.json", {"physical_properties_restored": True,
                        "construction_contact_normal": soft.contact_normal,
                        "physical_contact_normal": original.contact_normal,
                        "friction_history": "Fresh zero tangential springs",
                        "kinematics": "Positions/orientations retained; velocity and angular momentum reset to zero",
                        "diagnostics": engine.diagnostics()})
                phase_steps = 0
                while phase_steps < budget:
                    if shutil.disk_usage(out).free < p["minimum_free_bytes"]:
                        raise RuntimeError("Free-storage reserve reached")
                    if phase == "construction" and phase_steps % p["thermostat_interval"] == 0:
                        state = engine.state()
                        energy = cooling_energy(phase_steps, p["warm_steps"], p["cooling_steps"], p["agitation_energy_normalized"])
                        # After cooling, free damped relaxation with no further resets.
                        if phase_steps <= p["warm_steps"] + p["cooling_steps"]:
                            inertia = []
                            for i, scale, q in zip(ids, scales, state["rotation"]):
                                rotation = quat_matrix(q.astype(float) / np.linalg.norm(q))
                                body = library.shapes[int(i)].inertia * soft.density * scale**5 / (engine.mass_unit * engine.length_unit**2)
                                inertia.append(rotation @ body @ rotation.T)
                            before = engine.diagnostics()["kinetic_energy"]
                            v, angular = agitation(engine.body[:, 1], inertia, energy, rng)
                            for name, value in [("velocity", v), ("momentum", angular)]:
                                array = np.zeros((len(ids), 4), np.float32)
                                array[:, :3] = value
                                engine.buffers[name].copy_from_numpy(array)
                                engine.buffers[name + "Remainder"].copy_from_numpy(np.zeros_like(array))
                            thermostat_work += engine.diagnostics()["kinetic_energy"] - before
                    row = engine.diagnostics()
                    limits = engine.timestep_limits()
                    dt = select_timestep(limits["elastic_seconds"], limits["viscous_seconds"], p["timestep_cap_s"])
                    amount = min(p["batch_steps"], budget-phase_steps)
                    command = 0.
                    if phase == "physical-equilibration":
                        pressure = row["p"]
                        requested = np.clip(p["servo_gain"]*(pressure/p["target_pressure_Pa"]-1), -p["max_rate_s"], p["max_rate_s"])
                        # Compression-only overlap taper; expansion remains available.
                        taper = np.clip((p["hold_overlap"]-row["max_overlap_fraction"])/(p["hold_overlap"]-p["taper_overlap"]), 0., 1.)
                        command = float(max(requested, 0) + min(requested, 0)*taper)
                        command = float(np.clip(command, -p["max_batch_log_strain"]/(amount*dt), p["max_batch_log_strain"]/(amount*dt)))
                    for retry in range(p["maximum_timestep_halvings"]+1):
                        if dt < p["minimum_timestep_s"]:
                            raise RuntimeError("Adaptive timestep below frozen minimum")
                        engine.spec = replace(engine.spec, timestep=dt)
                        try:
                            engine.advance(amount, rate=[command]*3, drag=p["drag_per_step"]/dt)
                            break
                        except RejectedStep as exc:
                            rejections.append({"phase": phase, "step": total_steps, "dt": dt, "flags": exc.flags})
                            # Retry timestep/motion/penetration rejection at a smaller
                            # step only; unchanged guards, atomic rollback, bounded retries.
                            if exc.flags & ~(8192 | 4) or retry == p["maximum_timestep_halvings"]:
                                raise
                            dt *= .5
                    dt_values.append(dt)
                    total_steps += amount
                    phase_steps += amount
                    physical_time += amount*dt
                    row = engine.diagnostics()
                    row.update(phase=phase, phase_steps=phase_steps, total_steps=total_steps,
                               timestep_s=dt, elapsed_pseudo_time_s=physical_time,
                               wall_seconds=time.monotonic()-started, command_rate_s=command,
                               thermostat_work_J=thermostat_work)
                    rows.append(row)
                    log.write(json.dumps(row)+"\n")
                    log.flush()
                    write_json(out / "latest.json", row)
                    if total_steps % p["snapshot_steps"] == 0:
                        snapshot(engine, out / "state-latest.h5", phase, physical_time)
                        print(phase, total_steps, "phi", 1/(1+row["void_ratio"]), "overlap", row["max_overlap_fraction"], "p", row["p"], flush=True)
                    if phase == "physical-equilibration":
                        ok = (np.max(abs(np.diag(row["stress"])/p["target_pressure_Pa"]-1)) <= p["pressure_tolerance"]
                              and row["kinetic_energy"]/max(row["elastic_energy_proxy"], 1e-30) <= p["kinetic_tolerance"]
                              and row["unbalanced_force"] <= p["force_tolerance"]
                              and row["max_overlap_fraction"] <= original.max_overlap_fraction)
                        converged = converged+1 if ok else 0
                        if converged >= p["converged_samples"]:
                            break
                if phase == "physical-equilibration" and converged < p["converged_samples"]:
                    raise RuntimeError("Physical equilibrium not reached within frozen step budget")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        print(failure, flush=True)
    finally:
        if engine is not None:
            snapshot(engine, out / "state-final.h5", phase, physical_time)
        checks = {"physical_equilibrium": failure is None and converged >= p["converged_samples"],
                  "source_integrity": all(sha256(out / "source_snapshot" / name) == value for name, value in freeze["sources"].items()),
                  "input_integrity": all(sha256(path) == value for path, value in freeze["inputs"].items())}
        report = {"identity": p["identity"], "checks": checks, "passed": all(checks.values()),
                  "failure": failure, "phase": phase, "total_steps": total_steps,
                  "consecutive_equilibrium_samples": converged, "wall_seconds": time.monotonic()-started,
                  "dt_range_s": [min(dt_values), max(dt_values)] if dt_values else None,
                  "last": rows[-1] if rows else None, "rejections": rejections,
                  "scientific_admission": False,
                  "construction_contact_normal": soft.contact_normal,
                  "physical_contact_normal": original.contact_normal,
                  "scope": "Artificial mechanical construction and bounded equilibrium screen; not material validation, refinement qualification or ordinary lab checkpoint export"}
        write_json(out / "report.json", report)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        return worker(out)
    root = Path(__file__).resolve().parents[1]
    out.mkdir(parents=True, exist_ok=False)
    snap = out / "source_snapshot"
    shutil.copytree(root / "src/sand_atlas_dem", snap / "sand_atlas_dem", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, snap / Path(__file__).name)
    shutil.copy2(root / "tests/test_annealing.py", snap / "test_annealing.py")
    shutil.copy2(args.protocol, out / "protocol.json")
    p = json.loads(args.protocol.read_text())
    with h5py.File(p["seed_checkpoint"]) as f:
        shape = json.loads(f.attrs["experiment"])["shape_library"]
        assert sha256(shape) == f.attrs["shape_sha256"]
    inputs = {str(Path(path).resolve()): sha256(path) for path in [args.protocol, p["seed_checkpoint"], shape]}
    write_json(out / "prospective-freeze.json", {"inputs": inputs, "sources": {str(path.relative_to(snap)): sha256(path) for path in snap.rglob("*") if path.is_file()}, "scientific_admission": False})
    env = dict(os.environ, PYTHONPATH=str(snap), SAND_DEM_METAL="1", PYTHONUNBUFFERED="1")
    with (out / "run.log").open("w") as log:
        result = subprocess.run([sys.executable, str(snap / Path(__file__).name), str(out / "protocol.json"), "--output", str(out), "--worker"], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    write_json(out / "execution.json", {"returncode": result.returncode})
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
