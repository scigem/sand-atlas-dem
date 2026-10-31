"""Resolve an accepted adaptive reference into a frozen, full-inventory GPU study."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sand_atlas_dem.quadrature import prepare_adaptive, select_levels
from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute actual Metal after resolving all inputs",
    )
    args = parser.parse_args()
    policy = read_json(args.protocol)
    root = Path(__file__).resolve().parents[1]
    parent = Path(policy["physical_library"])
    reference_path = Path(policy["reference_report"])
    reference_protocol = reference_path.parent / "protocol.json"
    reference = read_json(reference_path)
    protocol = read_json(reference_protocol)
    # Fail before creating an output or preparing any asset unless all poses pass.
    selection = select_levels(
        reference, protocol, protocol["shape_ids"], target=policy["selection_target"]
    )
    if sha256(parent) != policy["parent_library_sha256"]:
        raise ValueError("Frozen physical library changed")
    if protocol["library_sha256"] != policy["parent_library_sha256"]:
        raise ValueError("Reference names different physical geometry")
    coarse = policy["libraries"][0]
    if len(policy["libraries"]) != 1 or coarse["subdivisions"] != 0:
        raise ValueError(
            "Prospective policy must specify exactly the coarse level-zero asset"
        )
    if sha256(coarse["path"]) != coarse["sha256"]:
        raise ValueError("Frozen coarse library changed")
    chosen = {r["shape"]: r["subdivisions"] for r in selection}
    maximum_active = max(
        row["levels"][chosen[row["shape"]]]["active_nodes"] for row in reference["rows"]
    )
    # Two bodies allocate twice this count. The factor is a declared resource
    # margin for FP32 boundary classification, never a force/count tolerance.
    contacts_per_particle = max(policy["minimum_contacts_per_particle"], maximum_active)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "private").mkdir(mode=0o700)
    paths = [
        Path(__file__),
        root / "examples/mesh_quadrature_study.py",
        args.protocol.resolve(),
        *sorted((root / "src/sand_atlas_dem").glob("*.py")),
        *sorted((root / "src/sand_atlas_dem/shaders").glob("*.slang")),
    ]
    hashes = {str(p.relative_to(root)): sha256(p) for p in paths}
    for path in paths:
        dest = output / "source_snapshot" / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    write_json(
        output / "prospective-freeze.json",
        dict(
            files=hashes,
            policy=policy,
            reference_report_sha256=sha256(reference_path),
            resource_policy=dict(
                largest_selected_reference_active_nodes=maximum_active,
                contacts_per_particle=contacts_per_particle,
                pair_contact_capacity=2 * contacts_per_particle,
            ),
        ),
    )
    receipt = prepare_adaptive(
        parent,
        reference_path,
        reference_protocol,
        output / "private/adaptive.h5",
        target=policy["selection_target"],
    )
    write_json(output / "preparation-receipt.json", receipt)
    gpu = {
        key: value
        for key, value in policy.items()
        if key
        not in (
            "physical_library",
            "selection_target",
            "minimum_contacts_per_particle",
            "capacity_policy",
        )
    }
    gpu.update(
        identity=policy["identity"] + "-resolved",
        reference_report=str(reference_path.resolve()),
        reference_report_sha256=sha256(reference_path),
        reference_protocol=str(reference_protocol.resolve()),
        reference_protocol_sha256=sha256(reference_protocol),
        reference_freeze_sha256=sha256(
            reference_path.parent / "prospective-freeze.json"
        ),
        contacts_per_particle=contacts_per_particle,
        libraries=[
            {**coarse, "path": str(Path(coarse["path"]).resolve())},
            dict(
                path=receipt["output"],
                sha256=receipt["sha256"],
                subdivisions_by_shape=chosen,
            ),
        ],
    )
    write_json(output / "gpu-protocol.json", gpu)
    if any(sha256(root / name) != digest for name, digest in hashes.items()):
        raise ValueError("Source changed during protocol resolution")
    write_json(
        output / "resolution.json",
        dict(
            status="ready",
            gpu_executed=False,
            gpu_protocol_sha256=sha256(output / "gpu-protocol.json"),
            reference_cases=len(reference["rows"]),
            expected_gpu_cases=2 * len(reference["rows"]),
            scientific_admission=False,
        ),
    )
    if not args.run:
        return 0
    with (output / "gpu.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(root / "examples/mesh_quadrature_study.py"),
                str(output / "gpu-protocol.json"),
                "--output",
                str(output / "gpu"),
            ],
            env={**os.environ, "PYTHONPATH": str(root / "src")},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    report = (
        read_json(output / "gpu/report.json")
        if (output / "gpu/report.json").exists()
        else {}
    )
    unchanged = all(sha256(root / name) == digest for name, digest in hashes.items())
    write_json(
        output / "report.json",
        dict(
            identity=policy["identity"],
            passed=bool(result.returncode == 0 and report.get("passed") and unchanged),
            source_unchanged=unchanged,
            scientific_admission=False,
            whole_library_static_only=True,
            completed_gpu_cases=len(report.get("rows", [])),
            expected_gpu_cases=2 * len(reference["rows"]),
        ),
    )
    return 0 if result.returncode == 0 and report.get("passed") and unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
