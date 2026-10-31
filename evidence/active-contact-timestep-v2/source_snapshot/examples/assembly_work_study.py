"""Frozen assembly work audit, executed only on the actual Metal backend."""

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.simulation import Simulation
from sand_atlas_dem.util import read_json, sha256, write_json


def energy(row):
    return row["kinetic_energy"] + row["elastic_energy_proxy"]


def power(row, rate):
    """Compression-positive Cauchy stress power for a diagonal physical cell."""
    return -float(np.prod(row["cell"]) * np.dot(np.diag(row["stress"]), rate))


def summarize(initial, observations):
    scale = energy(initial)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Positive initial assembly energy required")
    residuals, work_errors, dissipation = [], [], []
    integrated = 0.0
    previous = initial
    for row in observations:
        interval = row["study_time_s"] - previous.get("study_time_s", 0.0)
        if interval <= 0:
            raise ValueError("Strictly increasing study time required")
        rate = row["interval_rate_s"]
        integrated += 0.5 * (power(previous, rate) + power(row, rate)) * interval
        work = row["cell_work"] - initial["cell_work"]
        loss = sum(
            row[k] - initial[k] for k in ("friction_dissipation", "drag_dissipation")
        )
        residuals.append(abs(work - loss - (energy(row) - scale)) / scale)
        work_errors.append(abs(work - integrated) / scale)
        dissipation.extend(
            row[k] - previous[k] for k in ("friction_dissipation", "drag_dissipation")
        )
        previous = row
    values = np.array(residuals + work_errors + dissipation)
    if not len(observations) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty assembly observations required")
    return dict(
        initial_energy_J=scale,
        maximum_relative_proxy_balance_residual=float(max(residuals)),
        maximum_relative_independent_work_error=float(max(work_errors)),
        minimum_incremental_dissipation_J=float(min(dissipation)),
        independent_cell_work_J=integrated,
        final=row,
    )


def run_case(metal, policy, case, timestep, output):
    simulation = Simulation.resume(policy["prepared_checkpoint"], metal=metal)
    engine = simulation.engine
    # Each case is a new study initial-value problem, not an edited continuation.
    engine.spec = dataclasses.replace(engine.spec, timestep=timestep)
    initial = engine.diagnostics()
    if (
        engine.n != policy["particles"]
        or initial["active_grain_pairs"] < policy["minimum_active_grain_pairs"]
    ):
        raise ValueError("Prepared assembly inventory or contacts changed")
    rows = []
    write_json(output / "initial.json", initial)
    elapsed_steps = 0
    interval = round(policy["diagnostic_period_s"] / timestep)
    if interval < 1 or not np.isclose(
        interval * timestep, policy["diagnostic_period_s"], rtol=1e-12, atol=0
    ):
        raise ValueError("Diagnostic period must be an integer timestep multiple")
    for phase_index, phase in enumerate(case["phases"]):
        steps = round(phase["duration_s"] / timestep)
        if not np.isclose(steps * timestep, phase["duration_s"], rtol=1e-12, atol=0):
            raise ValueError("Phase duration must be an integer timestep multiple")
        for start in range(0, steps, interval):
            count = min(interval, steps - start)
            engine.advance(count, rate=phase["rate_s"], drag=phase["drag_s"])
            elapsed_steps += count
            row = dict(
                engine.diagnostics(),
                study_time_s=elapsed_steps * timestep,
                interval_rate_s=phase["rate_s"],
                phase=phase_index,
            )
            rows.append(row)
            with (output / "history.jsonl").open("a") as history:
                history.write(json.dumps(row) + "\n")
    metrics = summarize(initial, rows)
    metrics.update(
        case=case["name"],
        timestep_s=timestep,
        steps=elapsed_steps,
        minimum_active_grain_pairs=min(r["active_grain_pairs"] for r in rows),
        maximum_overlap_fraction=max(r["max_overlap_fraction"] for r in rows),
    )
    metrics["passed"] = bool(
        metrics["maximum_relative_proxy_balance_residual"]
        <= policy["maximum_relative_proxy_balance_residual"]
        and metrics["maximum_relative_independent_work_error"]
        <= policy["maximum_relative_independent_work_error"]
        and metrics["minimum_incremental_dissipation_J"]
        >= -policy["dissipation_roundoff_fraction"] * metrics["initial_energy_J"]
        and metrics["minimum_active_grain_pairs"]
        >= policy["minimum_active_grain_pairs"]
        and metrics["maximum_overlap_fraction"] <= policy["overlap_guard_fraction"]
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = read_json(args.protocol)
    inputs = {str(args.protocol): sha256(args.protocol)}
    for name in ("prepared_checkpoint", "acceptance_audit", "shape_library"):
        path = policy[name]
        if sha256(path) != policy[name + "_sha256"]:
            raise ValueError(f"Frozen {name} changed")
        inputs[path] = policy[name + "_sha256"]
    audit = read_json(policy["acceptance_audit"])
    if (
        not audit["passed"]
        or audit["prepared_checkpoint_sha256"] != policy["prepared_checkpoint_sha256"]
    ):
        raise ValueError("Preparation audit does not admit the initial state")
    with h5py.File(policy["prepared_checkpoint"]) as checkpoint:
        spec = json.loads(checkpoint.attrs["experiment"])
        if checkpoint.attrs["shape_sha256"] != policy["shape_library_sha256"] or spec[
            "shape_library"
        ] != str(Path(policy["shape_library"]).resolve()):
            raise ValueError("Initial checkpoint names different geometry")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        *sorted((root / "src/sand_atlas_dem").glob("*.py")),
        *sorted((root / "src/sand_atlas_dem/shaders").glob("*.slang")),
    ]
    sources = {str(p.relative_to(root)): sha256(p) for p in paths}
    for path in paths:
        target = args.output / "source_snapshot" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(args.protocol, args.output / "protocol.json")
    write_json(
        args.output / "prospective-freeze.json",
        dict(inputs=inputs, source_hashes=sources),
    )
    metal = Metal()
    rows, failures = [], []
    for case in policy["cases"]:
        for index, timestep in enumerate(policy["timesteps_s"]):
            output = args.output / "private" / f"{case['name']}-{index}"
            output.mkdir(parents=True, mode=0o700)
            try:
                row = run_case(metal, policy, case, timestep, output)
                rows.append(row)
                print(case["name"], timestep, row["passed"], flush=True)
            except Exception as exc:
                failures.append(
                    dict(
                        case=case["name"],
                        timestep_s=timestep,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                print(failures[-1], flush=True)
            write_json(
                args.output / "progress.json", dict(rows=rows, failures=failures)
            )
    refinement = []
    for case in policy["cases"]:
        selected = [r for r in rows if r["case"] == case["name"]]
        if len(selected) != len(policy["timesteps_s"]):
            continue
        middle, fine = selected[-2:]
        stress_error = float(
            np.linalg.norm(
                np.array(middle["final"]["stress"]) - fine["final"]["stress"]
            )
            / policy["initial_pressure_Pa"]
        )
        energy_error = (
            abs(energy(middle["final"]) - energy(fine["final"]))
            / fine["initial_energy_J"]
        )
        refinement.append(
            dict(
                case=case["name"],
                relative_stress_difference=stress_error,
                relative_energy_difference=energy_error,
                passed=stress_error <= policy["maximum_relative_refinement_difference"]
                and energy_error <= policy["maximum_relative_refinement_difference"],
            )
        )
    if any(sha256(path) != value for path, value in inputs.items()) or any(
        sha256(root / path) != value for path, value in sources.items()
    ):
        raise ValueError("Frozen study inputs or sources changed")
    complete = not failures and len(rows) == len(policy["cases"]) * len(
        policy["timesteps_s"]
    )
    passed = (
        complete
        and all(r["passed"] for r in rows)
        and len(refinement) == len(policy["cases"])
        and all(r["passed"] for r in refinement)
    )
    write_json(
        args.output / "report.json",
        dict(
            protocol=policy,
            inputs=inputs,
            source_hashes=sources,
            rows=rows,
            failures=failures,
            refinement=refinement,
            complete=complete,
            passed=passed,
            backend=metal.doctor(),
            scope=policy["scope"],
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
