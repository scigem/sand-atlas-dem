"""Freeze and screen experimental overlap-aware preparation on actual Metal."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from sand_atlas_dem.util import sha256, write_json


def clearance(e):
    state = e.state()
    pos = state["position"][:, :3].astype(float) + state["positionRemainder"][:, :3]
    cell = state["cell"][0, :3].astype(float)
    branch = pos[:, None, :] - pos[None, :, :]
    branch -= np.round(branch / cell) * cell
    distance = np.linalg.norm(branch, axis=-1)
    radii = e.body[:, 2].astype(float)
    np.fill_diagonal(distance, np.inf)
    return float(np.min(np.log(distance / (radii[:, None] + radii[None, :]))))


def fast_start(spec, policy, out):
    from sand_atlas_dem.preparation_control import fast_precontact_cap
    from sand_atlas_dem.simulation import Simulation

    sim = Simulation(spec)
    e = sim.engine
    initial = e.state()
    results = []
    for rate, steps in [(1000, 1000), (4000, 250)]:
        e.restore(initial, 0)
        begin = time.monotonic()
        commands = []
        for _ in range(steps // spec.diagnostic_interval):
            gap = clearance(e)
            cap = fast_precontact_cap(
                1000, rate, spec.diagnostic_interval * spec.timestep, gap
            )
            commands.append(dict(rate=cap, minimum_log_gap=gap))
            e.advance(
                spec.diagnostic_interval, rate=[-cap] * 3, drag=spec.stages[0]["drag"]
            )
        state = e.state()
        row = e.diagnostics()
        results.append(
            dict(
                rate=rate,
                steps=steps,
                commands=commands,
                wall_seconds=time.monotonic() - begin,
                diagnostics=row,
            )
        )
        results[-1]["endpoint"] = {
            "cell": state["cell"][0, :3].astype(float).tolist(),
            "position": (
                state["position"][:, :3].astype(float)
                + state["positionRemainder"][:, :3]
            ).tolist(),
            "velocity": (
                (
                    state["velocity"][:, :3].astype(float)
                    + state["velocityRemainder"][:, :3]
                )
                * e.length_unit
                / e.time_unit
            ).tolist(),
        }
    a, b = [r["endpoint"] for r in results]
    cell_error = float(np.max(abs(np.asarray(a["cell"]) / b["cell"] - 1)))
    position_error = float(np.max(abs(np.asarray(a["position"]) - b["position"])))
    checks = dict(
        no_contacts=all(
            r["diagnostics"]["active_surface_contacts"] == 0 for r in results
        ),
        equal_cell=cell_error <= policy["fast_cell_rtol"],
        equal_positions=position_error <= policy["fast_position_atol_normalized"],
        zero_peculiar_speed=all(
            np.linalg.norm(r["endpoint"]["velocity"], axis=1).max()
            <= policy["fast_velocity_atol_m_s"]
            for r in results
        ),
        clearance=all(c["minimum_log_gap"] > 0 for r in results for c in r["commands"]),
        fast_cap_used=all(
            abs(c["rate"] / 4000 - 1) <= 1e-12 for c in results[1]["commands"]
        ),
    )
    report = dict(
        cases=results,
        checks=checks,
        passed=all(checks.values()),
        cell_relative_error=cell_error,
        position_absolute_error_normalized=position_error,
        scope="Matched affine compression of a contact-free fixture; not full approach-to-contact or wall-clock throughput qualification.",
    )
    write_json(out / "fast-start.json", report)
    print("Fast-start checks", checks, flush=True)
    return report["passed"]


def execute(out):
    from sand_atlas_dem.spec import Experiment
    from sand_atlas_dem.preparation_control import OverlapControl
    from sand_atlas_dem.simulation import Simulation

    freeze = json.loads((out / "prospective-freeze.json").read_text())
    policy = json.loads((out / "protocol.json").read_text())
    assert sha256(out / "protocol.json") == freeze["protocol_sha256"]
    assert all(
        sha256(out / "source_snapshot" / p) == h for p, h in freeze["sources"].items()
    )
    assert all(sha256(p) == h for p, h in freeze["inputs"].items())
    with h5py.File(policy["parent_checkpoint"]) as f:
        spec = Experiment(**json.loads(f.attrs["experiment"]))
        parent_step = int(f.attrs["step"])
    assert spec.timestep == 2e-9 and spec.max_overlap_fraction == 0.01
    continuation = policy.get("mode") == "matched-density-continuation"
    fast_pass = None if continuation else fast_start(spec, policy, out)
    if not continuation and not fast_pass:
        raise RuntimeError(
            "Fast-start screen failed; preserve report and do not start contact trial"
        )
    sim = Simulation.resume(policy["parent_checkpoint"])
    original_rate = sim.rate
    restart_checks = {}
    if continuation:
        sidecar = json.loads(Path(policy["controller_state"]).read_text())
        restart_checks["checkpoint_binding"] = (
            sidecar["format"] == "sand-dem-overlap-control-state/1"
            and sidecar["checkpoint_sha256"] == sha256(policy["parent_checkpoint"])
        )
        restart_checks["same_policy"] = all(
            sidecar["control"][k] == v for k, v in policy["control"].items()
        )
        restored = sim.engine.state()
        with h5py.File(policy["parent_checkpoint"]) as f:
            if policy.get("restart_contract") == "keyed-derived-state/1":
                # Engine.restore rebuilds force/history storage at dt=0. Raw contact
                # order is not persistent identity; compare by immutable sample key.
                primary = policy["restart_primary_exact"]
                restart_checks["primary_state_exact"] = all(
                    np.array_equal(restored[k], f[k][:]) for k in primary
                )
                a = restored["meta"]
                b = f["meta"][:]
                ia = np.lexsort(a[:, :3].T[::-1])
                ib = np.lexsort(b[:, :3].T[::-1])
                keys_equal = np.array_equal(a[ia], b[ib])
                restart_checks["contact_keys_exact"] = keys_equal
                restart_checks["contact_count_exact"] = (
                    restored["contacts"] == int(f.attrs["contacts"])
                )
                restart_checks["keyed_normals_exact"] = keys_equal and np.array_equal(
                    restored["normalArea"][ia], f["normalArea"][:][ib]
                )
                restart_checks["keyed_slip_regression_bound"] = keys_equal and np.allclose(
                    restored["slip"][ia], f["slip"][:][ib],
                    rtol=policy["restart_slip_rtol"], atol=policy["restart_slip_atol"],
                )
                restart_checks["rebuilt_resultants_regression_bound"] = all(
                    np.allclose(restored[k], f[k][:],
                                rtol=policy["restart_resultant_rtol"],
                                atol=policy["restart_resultant_atol"])
                    for k in ["force", "torque", "accumulatedRemainder"]
                )
                write_json(out / "restart-differences.json", {
                    "keyed_slip_max_abs": float(np.max(abs(
                        restored["slip"][ia] - f["slip"][:][ib]
                    ))) if keys_equal else None,
                    "rebuilt_max_abs": {
                        k: float(np.max(abs(restored[k] - f[k][:])))
                        for k in ["force", "torque", "accumulatedRemainder"]
                    },
                })
            else:
                restart_checks["device_state_exact"] = all(
                    np.array_equal(restored[k], f[k][:]) for k in f
                ) and restored["contacts"] == int(f.attrs["contacts"])
            restart_checks["servo_state_exact"] = (
                sim.engine.step == int(f.attrs["step"])
                and sim.stage_index == int(f.attrs["stage_index"])
                and sim.stage_steps == int(f.attrs["stage_steps"])
                and sim.converged == int(f.attrs["converged"])
                and sim.contact_phase == bool(f.attrs["contact_phase"])
                and np.array_equal(sim.filter_stress, json.loads(f.attrs["filter_stress"]))
            )
        write_json(out / "restart-checks.json", restart_checks)
        if not all(restart_checks.values()):
            raise RuntimeError("Checkpoint/controller restore check failed before advance")
        control = OverlapControl(**sidecar["control"])
    else:
        control = OverlapControl(**policy["control"])
    with h5py.File(policy["old_failure_checkpoint"]) as f:
        old = json.loads(f.attrs["rows"])[-1]
    events = []
    counts = dict(hold_decisions=0, compress_decisions=0, hold_entries=0, hold_exits=0)
    parent = sim.rows[-1]
    started = time.monotonic()
    latest = out / "latest.json"
    with (out / "telemetry.jsonl").open("w") as log:

        def rate(stage, row):
            previous = control.held
            requested = original_rate(stage, row)
            command = control.command(requested, row["max_overlap_fraction"])
            counts["hold_decisions" if control.held else "compress_decisions"] += 1
            if previous != control.held:
                counts["hold_entries" if control.held else "hold_exits"] += 1
                events.append(
                    dict(
                        step=sim.engine.step,
                        held=control.held,
                        overlap=row["max_overlap_fraction"],
                    )
                )
            item = dict(
                step=sim.engine.step,
                physical_time_s=sim.engine.step * spec.timestep,
                held=control.held,
                requested_rate=requested.tolist(),
                command_rate=command.tolist(),
                max_overlap_fraction=row["max_overlap_fraction"],
                pressure_Pa=row["p"],
                solid_fraction=1 / (1 + row["void_ratio"]),
                counts=counts.copy(),
                wall_seconds=time.monotonic() - started,
            )
            log.write(json.dumps(item) + "\n")
            log.flush()
            write_json(latest, item)
            return command

        sim.rate = rate
        remaining = policy["steps"]
        part = 0
        result = None
        while remaining > 0:
            if continuation and result is not None and shutil.disk_usage(out).free < policy["minimum_free_bytes"]:
                print("Free-space reserve reached; stopping without deleting evidence", flush=True)
                break
            amount = min(remaining, policy["checkpoint_steps"])
            before_step = sim.engine.step
            result = sim.run(
                out / "run" / f"chunk-{part:04d}",
                max_chunks=amount // spec.diagnostic_interval,
            )
            remaining -= sim.engine.step - before_step
            checkpoint = result.path / "checkpoint.h5"
            write_json(
                result.path / "controller-state.json",
                dict(
                    format="sand-dem-overlap-control-state/1",
                    checkpoint_sha256=sha256(checkpoint)
                    if checkpoint.exists()
                    else None,
                    control=control.state(),
                    counts=counts,
                    events=events,
                ),
            )
            print("Checkpoint", sim.engine.step, result.status, counts, flush=True)
            part += 1
            if result.status != "paused":
                break
            if continuation and sim.rows[-1]["void_ratio"] <= old["void_ratio"]:
                print("Matched old failure density; stopping at checkpoint", flush=True)
                break
    final = sim.rows[-1]
    progress = (1 / (1 + final["void_ratio"])) / (1 / (1 + parent["void_ratio"])) - 1
    with h5py.File(policy["old_failure_checkpoint"]) as f:
        old = json.loads(f.attrs["rows"])[-1]
    observations = [r for r in sim.rows if r["step"] >= parent_step]
    checks = dict(
        fast_start=fast_pass,
        completed_window=sim.engine.step == parent_step + policy["steps"],
        crossed_old_failure_step=sim.engine.step > policy["old_failure_step"],
        guard=all(r["max_overlap_fraction"] <= 0.01 for r in observations),
        density_progress=progress >= policy["minimum_relative_density_increase"],
        recovered_from_hold=counts["hold_exits"] >= 1,
        source_integrity=all(
            sha256(out / "source_snapshot" / p) == h
            for p, h in freeze["sources"].items()
        ),
        input_integrity=all(sha256(p) == h for p, h in freeze["inputs"].items()),
    )
    if continuation:
        checks = dict(
            matched_density=final["void_ratio"] <= old["void_ratio"],
            execution=result.status in {"paused", "complete"},
            guard=checks["guard"],
            restart=all(restart_checks.values()),
            source_integrity=checks["source_integrity"],
            input_integrity=checks["input_integrity"],
        )
    report = dict(
        identity=policy["identity"],
        checks=checks,
        screen_passed=all(checks.values()),
        final_step=sim.engine.step,
        status=result.status,
        events=events,
        counts=counts,
        initial_solid_fraction=1 / (1 + parent["void_ratio"]),
        final_solid_fraction=1 / (1 + final["void_ratio"]),
        relative_density_increase=progress,
        old_failure_solid_fraction=1 / (1 + old["void_ratio"]),
        reached_old_failure_density=final["void_ratio"] <= old["void_ratio"],
        final_max_overlap=final["max_overlap_fraction"],
        final_pressure_Pa=final["p"],
        maximum_observed_overlap=max(r["max_overlap_fraction"] for r in observations),
        wall_seconds=time.monotonic() - started,
        completed_steps=sim.engine.step - parent_step,
        scientific_admission=False,
        scope=policy["scope"],
    )
    write_json(out / "report.json", report)
    return 0 if report["screen_passed"] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("protocol", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    out = a.output.resolve()
    if a.worker:
        return execute(out)
    root = Path(__file__).resolve().parents[1]
    out.mkdir(parents=True, exist_ok=False)
    snap = out / "source_snapshot"
    shutil.copytree(
        root / "src/sand_atlas_dem",
        snap / "sand_atlas_dem",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(__file__, snap / Path(__file__).name)
    shutil.copy2(
        root / "tests/test_preparation_control.py", snap / "test_preparation_control.py"
    )
    shutil.copy2(a.protocol, out / "protocol.json")
    policy = json.loads(a.protocol.read_text())
    inputs = {str(a.protocol.resolve()): sha256(a.protocol)}
    for key in ["parent_checkpoint", "old_failure_checkpoint"]:
        path = Path(policy[key])
        inputs[str(path)] = sha256(path)
        with h5py.File(path) as f:
            shape = Path(json.loads(f.attrs["experiment"])["shape_library"])
            assert sha256(shape) == f.attrs["shape_sha256"]
            inputs[str(shape)] = sha256(shape)
    if policy.get("controller_state"):
        path = Path(policy["controller_state"])
        inputs[str(path)] = sha256(path)
        sidecar = json.loads(path.read_text())
        assert sidecar["checkpoint_sha256"] == sha256(policy["parent_checkpoint"])
    write_json(
        out / "prospective-freeze.json",
        dict(
            identity=policy["identity"],
            inputs=inputs,
            protocol_sha256=sha256(out / "protocol.json"),
            sources={
                str(p.relative_to(snap)): sha256(p)
                for p in snap.rglob("*")
                if p.is_file()
            },
            scientific_admission=False,
        ),
    )
    env = dict(
        os.environ, PYTHONPATH=str(snap), SAND_DEM_METAL="1", PYTHONUNBUFFERED="1"
    )
    with (out / "run.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(snap / Path(__file__).name),
                str(out / "protocol.json"),
                "--output",
                str(out),
                "--worker",
            ],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    write_json(out / "execution.json", dict(returncode=result.returncode))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
