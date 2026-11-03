"""Plot completed, separately frozen serial/frontier throughput trials."""

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
    parser.add_argument("studies", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    inputs = {
        str((study / name).resolve()): sha256(study / name)
        for study in args.studies
        for name in [
            "report.json",
            "samples.json",
            "protocol.json",
            "prospective-freeze.json",
        ]
    }
    write_json(
        out / "analysis-freeze.json",
        dict(inputs=inputs, script_sha256=sha256(__file__)),
    )
    fig, axes = plt.subplots(
        1,
        len(args.studies),
        figsize=(5 * len(args.studies), 4.5),
        squeeze=False,
        layout="constrained",
    )
    reports = []
    for ax, study in zip(axes[0], args.studies):
        report = json.loads((study / "report.json").read_text())
        samples = json.loads((study / "samples.json").read_text())
        med = []
        ranges = {}
        for x, mode in enumerate(["serial", "parallel"]):
            values = np.array(
                [s["seconds_per_step"] * 1000 for s in samples if s["mode"] == mode]
            )
            med.append(float(np.median(values)))
            ranges[mode] = dict(
                min_ms=float(values.min()),
                max_ms=float(values.max()),
                median_ms=med[-1],
            )
            ax.scatter(
                np.full(len(values), x) + np.linspace(-0.06, 0.06, len(values)),
                values,
                color="#163e63",
                zorder=3,
                s=20,
            )
        ax.bar([0, 1], med, color=["#abc2d7", "#57a991"], width=0.65)
        ax.set_xticks([0, 1], ["Serial reference", "Parallel collector"])
        ax.set_ylim(0, max(med) * 1.2)
        ax.set_ylabel("Wall ms/step, fixed 2 ns timestep")
        ax.set_title(
            f"{study.name.rsplit('-v', 1)[-1]}: {report['speedup']:.2f}× speedup"
        )
        reports.append(
            dict(
                identity=report["identity"],
                speedup=report["speedup"],
                ranges=ranges,
                equivalence_passed=report["equivalence_passed"],
                performance_passed=report["performance_passed"],
            )
        )
    fig.suptitle("Hostun 424: six restored 128-step trials per collector")
    fig.savefig(out / "parallel-mesh-speedup.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    write_json(
        out / "summary.json",
        dict(
            studies=reports,
            inputs_unchanged=all(sha256(p) == h for p, h in inputs.items()),
            scope="Counterbalanced ordinary advancement wall times; setup/reset/warmup excluded. Dilute checkpoint, not dense throughput or GPU-only timings.",
        ),
    )


if __name__ == "__main__":
    main()
