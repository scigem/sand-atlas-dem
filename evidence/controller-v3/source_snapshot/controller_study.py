"""Run a frozen experiment in recoverable chunks; preserve every failed attempt."""

import argparse
import time
from pathlib import Path

from sand_atlas_dem.simulation import Simulation
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunks-per-checkpoint", type=int, default=100)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if args.chunks_per_checkpoint < 1:
        raise ValueError("Positive checkpoint interval required")
    args.output.mkdir(parents=True, exist_ok=False)
    spec = Experiment.load(args.experiment)
    sim = Simulation.resume(args.resume) if args.resume else Simulation(spec)
    # A resume uses the exact checkpoint experiment; do not relabel it as new input.
    if sim.experiment.to_dict() != spec.to_dict():
        raise ValueError("Checkpoint experiment differs from supplied protocol")
    start = time.perf_counter()
    chunks = 0
    write_json(
        args.output / "protocol.json",
        dict(
            experiment=spec.to_dict(),
            source_experiment_sha256=sha256(args.experiment),
            parent_checkpoint_sha256=sha256(args.resume) if args.resume else None,
        ),
    )
    while True:
        result = sim.run(
            args.output / f"chunk-{chunks:04d}", max_chunks=args.chunks_per_checkpoint
        )
        chunks += 1
        row = result.rows[-1]
        progress = dict(
            status=result.status,
            step=sim.engine.step,
            stage=sim.stage_index,
            converged=sim.converged,
            contact_phase=sim.contact_phase,
            last=row,
            elapsed_wall_seconds=time.perf_counter() - start,
            latest_checkpoint=str((result.path / "checkpoint.h5").resolve()),
        )
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
