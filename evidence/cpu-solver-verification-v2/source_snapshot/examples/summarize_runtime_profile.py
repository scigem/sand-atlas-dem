"""Summarize completed runtime profiling evidence without GPU-time inference."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sand_atlas_dem.util import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    inputs = [
        args.profile / p
        for p in [
            "report.json",
            "host-profile.json",
            "startup-profile.json",
            "batches.json",
            "split-stages.json",
            "prospective-freeze.json",
            "protocol.json",
        ]
    ]
    hashes = {str(p.resolve()): sha256(p) for p in inputs}
    write_json(
        out / "analysis-freeze.json",
        dict(inputs=hashes, script_sha256=sha256(__file__)),
    )
    r = json.loads((args.profile / "report.json").read_text())
    stages = r["stage_synchronized_wall"]
    names = list(stages)
    encode = np.array([stages[n]["encode_seconds"]["mean"] for n in names]) * 1000
    wait = np.array([stages[n]["submit_wait_seconds"]["mean"] for n in names]) * 1000
    total = encode + wait
    top = sorted(
        [
            dict(
                name=n,
                mean_wall_ms=float(t),
                encode_ms=float(a),
                submit_wait_ms=float(b),
                fraction_of_split_total=float(t / total.sum()),
            )
            for n, a, b, t in zip(names, encode, wait, total)
        ],
        key=lambda v: v["mean_wall_ms"],
        reverse=True,
    )
    write_json(
        out / "summary.json",
        dict(
            stage_ranking=top,
            split_encode_fraction=float(encode.sum() / total.sum()),
            batch_seconds_per_step=r["batch_seconds_per_step"],
            synchronized_total_ms=float(total.sum()),
            partition_check_passed=r["partition_check_passed"],
            inputs_unchanged=all(sha256(p) == h for p, h in hashes.items()),
            warning="Synchronized stage wall time includes driver/queue overhead and changes scheduling. It is not GPU execution time, and its sum is not unpartitioned throughput. Startup and Python call profiles include profiler overhead. Sparse checkpoint is not dense throughput.",
        ),
    )
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    ax.barh(names, encode, label="Encode (host wall)", color="#3a78ad")
    ax.barh(names, wait, left=encode, label="Submit + wait (wall)", color="#f2ab45")
    ax.set_xlabel("Mean ms / step, synchronized per subsystem")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    batches = r["batch_seconds_per_step"]
    keys = list(batches)
    bx.bar(keys, [batches[k]["median"] * 1000 for k in keys], color="#3a78ad")
    bx.set_xlabel("Steps per ordinary advance() call")
    bx.set_ylabel("Median wall ms / step")
    bx.set_title("Three restored repetitions / batch size")
    fig.suptitle("Current Hostun solver: 424 dilute grains, 2 ns fixed step")
    fig.text(
        0.5,
        -0.02,
        "Stage synchronization perturbs scheduling; neither panel reports GPU-only time.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(out / "runtime-profile.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
