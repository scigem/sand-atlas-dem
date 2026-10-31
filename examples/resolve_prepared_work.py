"""Resolve a prospective mesh-work study only from independently accepted preparation."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py

from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    policy = read_json(args.policy)
    preparation = Path(policy["requires_preparation"]).resolve()
    result = read_json(preparation / "report.json")
    audit_path = preparation / "acceptance-audit.json"
    audit = read_json(audit_path)
    if (
        not all(result.get(k) is True for k in ("passed", "source_unchanged"))
        or not audit.get("passed")
        or not audit.get("source_unchanged")
    ):
        raise ValueError("Terminal accepted preparation and independent audit required")
    resolved = {
        k: v
        for k, v in policy.items()
        if k
        not in (
            "requires_preparation",
            "required_contact_geometry",
            "required_source_shapes",
            "copies_per_shape",
        )
    }
    for key in ("prepared_checkpoint", "shape_library"):
        source = Path(audit[key]).resolve()
        if sha256(source) != audit[key + "_sha256"]:
            raise ValueError("Audited initial-state bytes changed")
        resolved[key] = str(source)
        resolved[key + "_sha256"] = audit[key + "_sha256"]
    resolved["acceptance_audit"] = str(audit_path)
    resolved["acceptance_audit_sha256"] = sha256(audit_path)
    if len(audit["shape_counts"]) != policy["required_source_shapes"] or set(
        audit["shape_counts"].values()
    ) != {policy["copies_per_shape"]}:
        raise ValueError("Exact full-library balanced inventory required")
    with h5py.File(resolved["prepared_checkpoint"]) as f:
        spec = json.loads(f.attrs["experiment"])
        if (
            spec["contact_geometry"] != policy["required_contact_geometry"]
            or spec["particles"] != policy["particles"]
            or spec["normal_viscosity"] != 0
        ):
            raise ValueError("Expected undamped mesh assembly contract required")
        if spec["stages"][0]["pressure"] != policy["initial_pressure_Pa"]:
            raise ValueError("Initial pressure differs from frozen work policy")
    root = Path(__file__).resolve().parents[1]
    frozen = read_json(preparation / "study/frozen-manifest.json")
    if (
        frozen.get("status") != "complete"
        or frozen.get("return_code") != 0
        or not frozen.get("sources")
    ):
        raise ValueError("Accepted immutable preparation source receipt required")
    for name, digest in audit["input_hashes"].items():
        if sha256(name) != digest:
            raise ValueError("Preparation audit input changed")
    if frozen["shape_sha256"] != resolved["shape_library_sha256"]:
        raise ValueError("Preparation source receipt names different geometry")
    archive = preparation / "study/source_snapshot"
    sources = {
        "examples/resolve_prepared_work.py": Path(__file__).resolve(),
        "examples/assembly_work_study.py": root / "examples/assembly_work_study.py",
    }
    for name, digest in frozen["sources"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe preparation source path")
        source = archive / relative
        if sha256(source) != digest:
            raise ValueError("Prepared-state source archive changed")
        target = "src/" + name if name.startswith("sand_atlas_dem/") else name
        if target in sources:
            raise ValueError("Preparation source collides with study driver")
        sources[target] = source
    required = {
        "src/sand_atlas_dem/" + name
        for name in (
            "__init__.py",
            "gpu.py",
            "simulation.py",
            "geometry.py",
            "spec.py",
            "backend.py",
            "shaders/solver.slang",
        )
    }
    if not required.issubset(sources):
        raise ValueError("Complete archived preparation solver required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, path in sources.items():
        dest = output / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[name] = sha256(dest)
    shutil.copy2(args.policy, output / "policy.json")
    shutil.copy2(preparation / "report.json", output / "preparation-report.json")
    shutil.copy2(audit_path, output / "acceptance-audit.json")
    write_json(output / "protocol.json", resolved)
    write_json(
        output / "prospective-freeze.json",
        dict(
            policy_sha256=sha256(args.policy),
            protocol_sha256=sha256(output / "protocol.json"),
            source_hashes=hashes,
            preparation_report_sha256=sha256(preparation / "report.json"),
            acceptance_audit_sha256=sha256(audit_path),
            scientific_admission=False,
            solver_source="Accepted preparation immutable archive; current checkout source is not substituted",
        ),
    )
    if not args.run:
        return 0
    with (output / "work.log").open("w") as log:
        completed = subprocess.run(
            [
                sys.executable,
                str(output / "source_snapshot/examples/assembly_work_study.py"),
                str(output / "protocol.json"),
                "--output",
                str(output / "study"),
            ],
            env={**os.environ, "PYTHONPATH": str(output / "source_snapshot/src")},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
