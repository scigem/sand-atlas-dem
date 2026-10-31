"""Consolidate a source-stable full reference and a verified same-pose extension.

No poses are added or removed. Every overlapping force must agree exactly.
The parent and supplement remain immutable evidence with distinct identities.
"""

import argparse
import copy
import shutil
from pathlib import Path

import numpy as np

from sand_atlas_dem.quadrature import select_levels
from sand_atlas_dem.util import read_json, sha256, write_json


def archive(path):
    report = read_json(path)
    protocol_path = path.parent / "protocol.json"
    freeze_path = path.parent / "prospective-freeze.json"
    freeze = read_json(freeze_path)
    if not report.get("passed") or not report.get("source_unchanged"):
        raise ValueError("Passing source-stable inputs required")
    if freeze["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Changed archived protocol")
    sources = {}
    for name, digest in freeze["source_hashes"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source archive path")
        source = path.parent / "source_snapshot" / relative
        if sha256(source) != digest:
            raise ValueError("Changed archived source")
        sources[name] = source
    return report, read_json(protocol_path), sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument("policy", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    policy = read_json(args.policy)
    paths = [Path(policy[k]) for k in ("parent_report", "supplement_report")]
    for key, path in zip(("parent_report", "supplement_report"), paths):
        if sha256(path) != policy[key + "_sha256"]:
            raise ValueError("Changed prospectively named report")
    parent, protocol, sources = archive(paths[0])
    supplement, extra_protocol, extra_sources = archive(paths[1])
    if not parent.get("reference_refinement_passed") or not all(
        supplement.get(k) for k in ("refinement_passed", "overlap_passed")
    ):
        raise ValueError("Unaccepted reference input")
    if protocol["library_sha256"] != extra_protocol["library_sha256"]:
        raise ValueError("Different physical geometry")
    row = copy.deepcopy(supplement["row"])
    matches = [
        i
        for i, r in enumerate(parent["rows"])
        if (r["shape"], r["target_sampled_depth_fraction"])
        == (row["shape"], row["target_sampled_depth_fraction"])
    ]
    if len(matches) != 1:
        raise ValueError("Supplement must extend one existing unique pose")
    old = parent["rows"][matches[0]]
    if row["separation_m"] != old["separation_m"] or len(row["levels"]) <= len(
        old["levels"]
    ):
        raise ValueError("Supplement changed pose or did not extend it")
    for a, b in zip(old["levels"], row["levels"]):
        if a["level"] != b["level"] or not np.array_equal(
            a["force_integral_m3"], b["force_integral_m3"]
        ):
            raise ValueError("Overlapping force changed")
    row["reference_finest_level"] = row["levels"][-1]["level"]
    row["level4_relative_change"] = old["level4_relative_change"]
    report = copy.deepcopy(parent)
    report["identity"] = policy["identity"]
    report["rows"][matches[0]] = row
    # The existing complete-inventory selector rechecks all 106 rows and gates.
    selection_error = None
    try:
        selection = select_levels(report, protocol, protocol["shape_ids"], target=0.05)
    except ValueError as exc:
        # All reference/inventory gates run before production-level selection.
        # A production capability gap must not erase valid finer reference data.
        if not str(exc).startswith("No independently supported quadrature level for "):
            raise
        selection = None
        selection_error = str(exc)
    numerical_dependencies = [
        "examples/library_quadrature_deterministic.py",
        "examples/triangle_quadrature_minimum.py",
        "examples/reference_inside.py",
        "examples/reference_rays.py",
        "examples/reference_winding.py",
        "src/sand_atlas_dem/geometry.py",
        "src/sand_atlas_dem/reference.py",
        "src/sand_atlas_dem/util.py",
    ]
    for name in numerical_dependencies:
        matching = [
            path
            for key, path in sources.items()
            if key == name or key.endswith("/" + name)
        ]
        if not matching or any(
            sha256(path) != sha256(extra_sources[name]) for path in matching
        ):
            raise ValueError("Numerical reference operator changed")
    # Preserve both complete archives: unrelated CLI/controller files may differ.
    sources = {"parent/" + name: path for name, path in sources.items()} | {
        "supplement/" + name: path for name, path in extra_sources.items()
    }
    sources["examples/consolidate_reference.py"] = Path(__file__)
    sources["src/sand_atlas_dem/quadrature.py"] = (
        Path(__file__).resolve().parents[1] / "src/sand_atlas_dem/quadrature.py"
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, source in sources.items():
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = sha256(target)
    protocol.update(
        identity=policy["identity"],
        consolidation_policy=policy,
        scope="All 106 original poses retained; one original pose extended to a strictly finer reference, preserving every overlapping force exactly. Parent and supplement reports/archives retained. No production, dynamic or material admission.",
    )
    write_json(output / "protocol.json", protocol)
    write_json(
        output / "prospective-freeze.json",
        dict(
            protocol_sha256=sha256(output / "protocol.json"),
            source_hashes=hashes,
            input_policy_sha256=sha256(args.policy),
        ),
    )
    shutil.copy2(args.policy, output / "consolidation-policy.json")
    for name, source in zip(("parent-report.json", "supplement-report.json"), paths):
        shutil.copy2(source, output / name)
    report.update(
        parent_report_sha256=sha256(paths[0]),
        supplement_report_sha256=sha256(paths[1]),
        scope=protocol["scope"],
        exact_overlapping_force_agreement=True,
        elapsed_seconds=None,
        production_selection_supported=selection is not None,
        production_selection_error=selection_error,
    )
    write_json(output / "selection.json", selection)
    write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
