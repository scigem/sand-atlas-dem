"""Frozen short controller/hold replays of retained Hostun preparation checkpoints."""

import argparse
import copy
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


def worst_contact(engine):
    count = int(engine.check_errors()[2])
    if not count:
        return None
    keys = engine.m.read_prefix(
        engine.history_buffers[1 - engine.active]["meta"], 4 * count, np.uint32, shape=(count, 4)
    )
    depths = engine.m.read_prefix(
        engine.buffers["pointDepth"], 4 * count, shape=(count, 4)
    )[:, 3]
    body = engine.m.read(engine.buffers["body"], shape=(engine.n, 4))
    ratios = depths / np.minimum(body[keys[:, 0], 3], body[keys[:, 1], 3])
    index = int(np.argmax(ratios))
    src, dst, node = map(int, keys[index, :3])
    return dict(
        source_body=src,
        target_body=dst,
        source_node=node,
        source_shape=engine.library.shapes[int(engine.ids[src])].id,
        target_shape=engine.library.shapes[int(engine.ids[dst])].id,
        depth_m=float(depths[index] * engine.length_unit),
        overlap_fraction=float(ratios[index]),
    )


def execute(out):
    from sand_atlas_dem.gpu import RejectedStep
    from sand_atlas_dem.simulation import Simulation

    frozen = json.loads((out / "prospective-freeze.json").read_text())
    policy = json.loads((out / "protocol.json").read_text())
    assert sha256(out / "protocol.json") == frozen["protocol_sha256"]
    assert all(
        sha256(out / "source_snapshot" / p) == h for p, h in frozen["sources"].items()
    )
    assert all(sha256(p) == h for p, h in frozen["inputs"].items())
    report = dict(
        identity=policy["identity"],
        backend="Metal",
        cases=[],
        scientific_admission=False,
    )
    for item in policy["checkpoints"]:
        with h5py.File(item["path"]) as f:
            state = {name: f[name][:] for name in f}
            state["contacts"] = int(f.attrs["contacts"])
            parent_step = int(f.attrs["step"])
            filter_stress = np.asarray(json.loads(f.attrs["filter_stress"]))
        print("Loading", item["name"], parent_step, flush=True)
        sim = Simulation.resume(item["path"])
        engine = sim.engine
        assert engine.spec.timestep == 2e-9 and engine.spec.max_overlap_fraction == 0.01
        assert not engine.viscous and not engine.wall_count
        for branch in policy["branches"]:
            name = item["name"] + "-" + branch["name"]
            case = dict(
                name=name,
                parent_step=parent_step,
                mode=branch,
                requested_steps=item["steps"],
                status="running",
                rows=[],
            )
            report["cases"].append(case)
            engine.restore(state, parent_step)
            sim.filter_stress = filter_stress.copy()
            sim.contact_phase = True
            stage = copy.deepcopy(sim.experiment.stages[0])
            assert stage["drag"] == policy["drag"]
            if branch["cap_s_inv"]:
                stage["contact_max_rate"] = branch["cap_s_inv"]
            first = engine.diagnostics()
            first["worst_contact"] = worst_contact(engine)
            case["rows"].append(first)
            start = time.monotonic()
            print(
                "Starting", name, "overlap", first["max_overlap_fraction"], flush=True
            )
            for _ in range(item["steps"] // policy["diagnostic_interval"]):
                row = case["rows"][-1]
                rate = sim.rate(stage, row) if branch["cap_s_inv"] else np.zeros(3)
                try:
                    engine.advance(
                        policy["diagnostic_interval"], rate=rate, drag=policy["drag"]
                    )
                except RejectedStep as error:
                    case.update(
                        status="rejected",
                        failure=str(error),
                        flags=int(error.flags),
                        rejected_interval=[
                            engine.step,
                            engine.step + policy["diagnostic_interval"],
                        ],
                    )
                    break
                row = engine.diagnostics()
                row["worst_contact"] = worst_contact(engine)
                case["rows"].append(row)
                write_json(out / "report.json", report)
                print(
                    name, row["step"], row["max_overlap_fraction"], row["p"], flush=True
                )
            else:
                case["status"] = "completed_under_guard"
            final = case["rows"][-1]
            case.update(
                completed_steps=engine.step - parent_step,
                wall_seconds=time.monotonic() - start,
                overlap_change=final["max_overlap_fraction"]
                - first["max_overlap_fraction"],
                cell_relative_change=(
                    np.asarray(final["cell"]) / first["cell"] - 1
                ).tolist(),
            )
            # Preserve the last valid state; rejected batches roll back transactionally.
            sim.rows = case["rows"]
            sim.stage_steps = engine.step
            sim.checkpoint(out / "private" / (name + ".h5"))
            write_json(out / "report.json", report)
            print("Finished", name, case["status"], flush=True)
        del sim, engine
    report["source_integrity"] = all(
        sha256(out / "source_snapshot" / p) == h for p, h in frozen["sources"].items()
    )
    report["input_integrity"] = all(sha256(p) == h for p, h in frozen["inputs"].items())
    report["complete"] = True
    write_json(out / "report.json", report)
    return 0 if report["source_integrity"] and report["input_integrity"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        return execute(out)
    root = Path(__file__).resolve().parents[1]
    out.mkdir(parents=True, exist_ok=False)
    (out / "private").mkdir()
    snapshot = out / "source_snapshot"
    shutil.copytree(
        root / "src/sand_atlas_dem",
        snapshot / "sand_atlas_dem",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    shutil.copy2(args.protocol, out / "protocol.json")
    policy = json.loads(args.protocol.read_text())
    inputs = {str(args.protocol.resolve()): sha256(args.protocol)}
    for item in policy["checkpoints"]:
        p = Path(item["path"])
        inputs[str(p)] = sha256(p)
        with h5py.File(p) as f:
            shape = Path(json.loads(f.attrs["experiment"])["shape_library"])
            assert sha256(shape) == f.attrs["shape_sha256"]
        inputs[str(shape)] = sha256(shape)
    write_json(
        out / "prospective-freeze.json",
        dict(
            identity=policy["identity"],
            protocol_sha256=sha256(out / "protocol.json"),
            inputs=inputs,
            sources={
                str(p.relative_to(snapshot)): sha256(p)
                for p in snapshot.rglob("*")
                if p.is_file()
            },
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            scientific_admission=False,
        ),
    )
    env = dict(
        os.environ, PYTHONPATH=str(snapshot), SAND_DEM_METAL="1", PYTHONUNBUFFERED="1"
    )
    with (out / "run.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(snapshot / Path(__file__).name),
                str(out / "protocol.json"),
                "--output",
                str(out),
                "--worker",
            ],
            env=env,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    write_json(out / "execution.json", dict(returncode=result.returncode))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
