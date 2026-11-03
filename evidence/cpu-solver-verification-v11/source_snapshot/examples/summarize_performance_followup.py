"""Audit immutable follow-up studies and plot diagnostic stage-wall costs."""

import argparse
import json
import shutil
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
    inputs = {}
    for study in args.studies:
        for name in [
            "report.json",
            "failure-receipt.json",
            "protocol.json",
            "prospective-freeze.json",
            "samples.json",
            "split-stages.json",
        ]:
            p = study / name
            if p.exists():
                inputs[str(p.resolve())] = sha256(p)
    write_json(
        out / "analysis-freeze.json",
        dict(inputs=inputs, script_sha256=sha256(__file__)),
    )
    shutil.copyfile(__file__, out / Path(__file__).name)
    checks = []
    reports = []
    for study in args.studies:
        freeze = json.loads((study / "prospective-freeze.json").read_text())
        hashes = freeze.get("sources", freeze.get("files", {}))
        archived = all(
            sha256(study / "source_snapshot" / n) == h for n, h in hashes.items()
        )
        original = all(sha256(n) == h for n, h in freeze.get("inputs", {}).items())
        if (study / "protocol.json").exists():
            policy = json.loads((study / "protocol.json").read_text())
            for path_key, hash_key in [
                ("shape_library", "shape_sha256"),
                ("checkpoint", "checkpoint_sha256"),
                ("reference_shader", "reference_shader_sha256"),
            ]:
                if path_key in policy:
                    original &= sha256(policy[path_key]) == policy[hash_key]
        protocol = (
            "protocol_sha256" not in freeze
            or sha256(study / "protocol.json") == freeze["protocol_sha256"]
        )
        checks.append(
            dict(
                study=study.name,
                archived_files=len(hashes),
                archive_verified=archived,
                original_inputs_verified=original,
                protocol_verified=protocol,
            )
        )
        assert archived and original and protocol
        if (study / "report.json").exists():
            r = json.loads((study / "report.json").read_text())
            reports.append((study.name, r))
    profiles = [(n, r) for n, r in reports if "stage_mean_seconds" in r]
    fig, ax = plt.subplots(figsize=(10, 4.7), layout="constrained")
    stages = [
        "drift",
        "neighbours",
        "contact_geometry",
        "friction_and_history",
        "force_reduction",
        "final_kick",
    ]
    labels = [
        "Drift",
        "Neighbours",
        "Contact geometry",
        "Friction/history",
        "Force gathering",
        "Final kick",
    ]
    colors = ["#b6c8d9", "#8ea8bf", "#3c8097", "#b6c885", "#e8a14d", "#c6b0cf"]
    left = np.zeros(len(profiles))
    names = []
    for n, r in profiles:
        names.append(
            (
                "Hostun424 sparse"
                if n.startswith("hostun")
                else "Dense27 " + ("spinning" if "gather" in n else "zero-spin")
            )
            + (
                "\n(diagnostic; equivalence failed)"
                if not r["equivalence_passed"]
                else ""
            )
        )
    for key, label, color in zip(stages, labels, colors):
        values = np.array([r["stage_mean_seconds"][key] * 1000 for _, r in profiles])
        ax.barh(names, values, left=left, label=label, color=color)
        left += values
    ax.set_xlabel("Mean synchronized stage wall ms/step (not GPU timestamps)")
    ax.set_title("Post-optimization bottlenecks depend on contact density")
    ax.invert_yaxis()
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(out / "stage-wall-comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    unchanged = all(sha256(n) == h for n, h in inputs.items())
    write_json(
        out / "archive-check.json",
        dict(studies=checks, analysis_inputs_unchanged=unchanged),
    )
    assert unchanged
    write_json(
        out / "summary.json",
        dict(
            studies=[
                dict(
                    identity=n,
                    completed=r.get("completed"),
                    equivalence_passed=r.get("equivalence_passed"),
                    performance_passed=r.get("performance_passed"),
                    speedup=r.get("speedup"),
                    speedups=r.get("speedups"),
                    stage_mean_seconds=r.get("stage_mean_seconds"),
                )
                for n, r in reports
            ],
            scope="Diagnostic wall profiles; failed studies remain failed. No material admission or GPU-only timestamp claim.",
        ),
    )


if __name__ == "__main__":
    main()
