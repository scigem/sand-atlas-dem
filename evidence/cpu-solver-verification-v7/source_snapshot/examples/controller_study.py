"""Run a frozen experiment in recoverable chunks; preserve every failed attempt."""

import argparse
import time
from pathlib import Path

from sand_atlas_dem.simulation import Simulation
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import sha256, write_json


def reconcile_resume(existing, requested, *, extend_step_budgets=False):
    """Preserve grown allocations; only explicit budget extensions may change stages."""
    old = existing.to_dict()
    new = requested.to_dict()
    result = requested.to_dict()
    for key in ("contacts_per_particle", "pairs_per_particle", "bin_capacity"):
        if old[key] < new[key]:
            raise ValueError("Resume allocation is smaller than requested protocol")
        result[key] = old[key]
        old.pop(key)
        new.pop(key)
    # Physical input identity is verified separately against exact file hashes.
    old.pop("shape_library")
    new.pop("shape_library")
    if extend_step_budgets:
        if len(old["stages"]) != len(new["stages"]):
            raise ValueError("Budget extension cannot add or remove stages")
        for before, after in zip(old["stages"], new["stages"]):
            if after["steps"] < before["steps"]:
                raise ValueError("Budget extension cannot shorten a stage")
            before.pop("steps")
            after.pop("steps")
    if old != new:
        raise ValueError("Checkpoint physical protocol differs from supplied protocol")
    return Experiment(**result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunks-per-checkpoint", type=int, default=100)
    p.add_argument("--resume", type=Path)
    p.add_argument("--extend-step-budgets", action="store_true")
    args = p.parse_args()
    if args.chunks_per_checkpoint < 1:
        raise ValueError("Positive checkpoint interval required")
    args.output.mkdir(parents=True, exist_ok=False)
    spec = Experiment.load(args.experiment)
    sim = Simulation.resume(args.resume) if args.resume else Simulation(spec)
    original_experiment = sim.experiment.to_dict()
    if args.extend_step_budgets and args.resume is None:
        raise ValueError("A budget extension requires a parent checkpoint")
    if args.resume:
        if sha256(Path(sim.experiment.shape_library)) != sha256(
            Path(spec.shape_library)
        ):
            raise ValueError("Checkpoint shape bytes differ from supplied protocol")
        reconciled = reconcile_resume(
            sim.experiment, spec, extend_step_budgets=args.extend_step_budgets
        )
        sim.experiment = reconciled
        sim.engine.spec = reconciled
    start = time.perf_counter()
    chunks = 0
    write_json(
        args.output / "protocol.json",
        {
            "experiment": spec.to_dict(),
            "source_experiment_sha256": sha256(args.experiment),
            "parent_checkpoint_sha256": sha256(args.resume) if args.resume else None,
            "parent_experiment": original_experiment if args.resume else None,
            "effective_experiment": sim.experiment.to_dict(),
            "explicit_step_budget_extension": args.extend_step_budgets,
        },
    )
    while True:
        result = sim.run(
            args.output / f"chunk-{chunks:04d}", max_chunks=args.chunks_per_checkpoint
        )
        chunks += 1
        row = result.rows[-1]
        progress = {
            "status": result.status,
            "step": sim.engine.step,
            "stage": sim.stage_index,
            "converged": sim.converged,
            "contact_phase": sim.contact_phase,
            "last": row,
            "elapsed_wall_seconds": time.perf_counter() - start,
            "latest_checkpoint": str((result.path / "checkpoint.h5").resolve()),
        }
        write_json(args.output / "progress.json", progress)
        print(
            sim.engine.step,
            result.status,
            row["p"],
            row["unbalanced_force"],
            row["kinetic_energy"] / max(row["elastic_energy_proxy"], 1e-30),
            flush=True,
        )
        if result.status != "paused":
            break
    write_json(args.output / "result.json", progress)
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
