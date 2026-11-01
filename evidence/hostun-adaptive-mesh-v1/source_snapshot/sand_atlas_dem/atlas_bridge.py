"""Explicit synthetic DEM projection into Atlas's ordered replay interface."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from .spec import Experiment
from .util import finite_array, read_json, sha256, write_json

POLICY = {
    "format": "sand-dem-atlas-projection/1",
    "stress": "axisymmetric-mean-radial-projection",
    "strain": "prepared-state-engineering",
    "effective_stress": "dry-no-pore-pressure",
    "kinematics": "caller-accepted-infinitesimal-approximation",
}
PERMUTATION = [2, 0, 1]  # Right-handed DEM z,x,y -> Atlas x,y,z.


def _snapshot(path):
    """Hash the exact parsed bytes and reject ambiguous or nonfinite records."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    content = Path(path).read_bytes()
    value = json.loads(
        content,
        object_pairs_hook=unique,
        parse_constant=lambda s: (_ for _ in ()).throw(
            ValueError(f"Nonfinite JSON: {s}")
        ),
    )
    return value, hashlib.sha256(content).hexdigest()


def validate_policy(policy):
    """Validate caller declarations before simulation or conversion."""
    if set(policy) != {*POLICY, "maximum_component_strain", "note"} or any(
        policy[k] != v for k, v in POLICY.items()
    ):
        raise ValueError("Explicit sand-dem-atlas-projection/1 declarations required")
    bound = policy["maximum_component_strain"]
    if type(bound) not in (int, float) or not np.isfinite(bound) or not 0 < bound < 0.5:
        raise ValueError(
            "Explicit component-strain bound between zero and 0.5 required"
        )
    if not isinstance(policy["note"], str) or not policy["note"].strip():
        raise ValueError("Explain the caller's kinematic approximation")
    return bound


def project(manifest, rows, prepared, policy):
    """Project a single accepted preparation/triaxial case without filtering rows."""
    bound = validate_policy(policy)
    if (
        manifest.get("format") != "sand-dem-run/1"
        or manifest.get("kind") != "synthetic-dem"
        or manifest.get("status") != "complete"
        or manifest.get("stage_index") != 2
    ):
        raise ValueError("Only completed synthetic DEM runs can be handed off")
    if manifest.get("units") != dict(length="m", stress="Pa", time="s", strain="1"):
        raise ValueError("Unsupported DEM units")
    conventions = manifest.get("conventions", {})
    if (
        conventions.get("stress") != "compression_positive"
        or conventions.get("tensor_order") != "xx,xy,xz;yx,yy,yz;zx,zy,zz"
        or conventions.get("stress_measure") != "contact_virial_plus_peculiar_kinetic"
    ):
        raise ValueError("Unsupported DEM stress conventions")
    stages = manifest["experiment"]["stages"]
    if [s["kind"] for s in stages] != ["isotropic", "triaxial"]:
        raise ValueError(
            "Expected one isotropic preparation followed by one triaxial stage"
        )
    if not rows or any(
        type(r.get("stage")) is not int or r["stage"] not in (0, 1) for r in rows
    ):
        raise ValueError("Invalid source stages")
    if any(
        type(r.get("step")) is not int
        or r["step"] < 0
        or type(r.get("elapsed_time")) not in (int, float)
        or not np.isfinite(r["elapsed_time"])
        or r["elapsed_time"] < 0
        for r in rows
    ):
        raise ValueError("Finite nonnegative source steps and times required")
    if any(
        rows[i]["step"] >= rows[i + 1]["step"]
        or rows[i]["elapsed_time"] >= rows[i + 1]["elapsed_time"]
        or rows[i]["stage"] > rows[i + 1]["stage"]
        for i in range(len(rows) - 1)
    ):
        raise ValueError("Source step/time/stage order must be preserved")
    prep = [r for r in rows if r["stage"] == 0]
    loading = [r for r in rows if r["stage"] == 1]
    if len(prep) < 100 or not loading or len(loading) > 5000:
        raise ValueError(
            "Require 100 preparation diagnostics and 1–5000 loading rows; no automatic downsampling"
        )
    origin = prep[-1]
    initial_cell = finite_array(origin["cell"], (3,), positive=True)
    if (
        prepared["stage_index"] != 1
        or prepared["stage_steps"] != 0
        or prepared["step"] != origin["step"]
        or not np.array_equal(
            finite_array(prepared["initial_cell"], (3,)), initial_cell
        )
    ):
        raise ValueError("Prepared checkpoint does not identify the loading origin")
    pressure = stages[0]["pressure"]
    if not np.isfinite(pressure) or pressure <= 0:
        raise ValueError("Positive preparation pressure required")
    # These project gates are fixed before reading a case; caller declarations
    # cannot relax the 1% / 1e-5 / 1e-3 / 100-diagnostic acceptance policy.
    for r in prep[-100:]:
        stress = finite_array(r["stress"], (3, 3))
        kinetic, elastic, force = (
            r["kinetic_energy"],
            r["elastic_energy_proxy"],
            r["unbalanced_force"],
        )
        if (
            not np.isfinite([kinetic, elastic, force]).all()
            or min(kinetic, elastic, force) < 0
            or np.max(np.abs(np.diag(stress) / pressure - 1)) > 0.01
            or kinetic / max(elastic, 1e-30) > 1e-5
            or force > 1e-3
        ):
            raise ValueError(
                "Preparation fails the independent 100-diagnostic acceptance audit"
            )
    observations, details = [], []
    for row in [origin, *loading]:
        cell = finite_array(row["cell"], (3,), positive=True)
        ratio = cell / initial_cell
        strain = 1 - ratio
        if np.abs(strain).max() > bound:
            raise ValueError(
                "Case exceeds the caller's declared strain-approximation bound"
            )
        stress = finite_array(row["stress"], (3, 3))
        radial = float((stress[0, 0] + stress[1, 1]) / 2)
        axial = float(stress[2, 2])
        e = row["void_ratio"]
        if not np.isfinite(e) or e <= -1:
            raise ValueError("Invalid nominal void ratio")
        observations.append(
            dict(
                axial_strain=float(strain[2]),
                radial_strain=float((strain[0] + strain[1]) / 2),
                volumetric_strain=float(strain.sum()),
                p=float(np.trace(stress) / 3),
                q=axial - radial,
                axial_effective_stress=axial,
                radial_effective_stress=radial,
                void_ratio=e,
                elapsed_time=float(row["elapsed_time"] - origin["elapsed_time"]),
            )
        )
        symmetric = (stress + stress.T) / 2
        dev = symmetric - np.trace(symmetric) / 3 * np.eye(3)
        details.append(
            dict(
                source_step=row["step"],
                source_stage=row["stage"],
                raw=copy.deepcopy(row),
                stress_tensor_atlas=stress[np.ix_(PERMUTATION, PERMUTATION)],
                engineering_strain_atlas=strain[PERMUTATION],
                logarithmic_strain_atlas=(-np.log(ratio))[PERMUTATION],
                finite_volume_strain=float(1 - np.prod(ratio)),
                q_j2=float(np.sqrt(1.5) * np.linalg.norm(dev)),
                radial_stress_difference_pa=float(stress[0, 0] - stress[1, 1]),
                symmetric_shear_norm_pa=float(
                    np.linalg.norm(symmetric - np.diag(np.diag(symmetric)))
                ),
                stress_antisymmetry_norm_pa=float(
                    np.linalg.norm((stress - stress.T) / 2)
                ),
            )
        )
    first = observations[0]
    return dict(
        format="sand-dem-atlas-virtual-test/1",
        kind="synthetic-dem",
        protocol="drained-triaxial",
        scientific_admission=False,
        policy=copy.deepcopy(policy),
        conventions=dict(
            stress="compression-positive",
            strain="compression-positive",
            stress_unit="Pa",
            strain_unit="1",
            axial_mapping="DEM z -> Atlas x; DEM x,y -> Atlas y,z",
            strain_measure="prepared-state-engineering",
            volume_strain_measure="sum-of-engineering-components",
            pore_pressure="absent-dry-grain-model",
            q="signed-axial-minus-mean-radial",
        ),
        initial=dict(
            effective_stress=[
                first["axial_effective_stress"],
                first["radial_effective_stress"],
                first["radial_effective_stress"],
                0,
                0,
                0,
            ],
            void_ratio=first["void_ratio"],
        ),
        preparation=dict(
            source_rows=len(prep),
            accepted_diagnostics=100,
            origin_step=origin["step"],
            pressure_pa=pressure,
            pressure_relative_tolerance=0.01,
            kinetic_elastic_ratio_tolerance=1e-5,
            force_tolerance=1e-3,
        ),
        rows=observations,
        dem_observations=details,
        warnings=[
            "Synthetic DEM virtual test; not measured experimental evidence or material calibration.",
            "Axisymmetric replay uses mean radial normal stress and discards shear/radial asymmetry; full tensors are retained.",
            "Engineering strain is mapped to an infinitesimal continuum path only under the explicit caller declaration.",
            "Nominal void ratio ignores overlap volume. Physical elapsed time is not used by incremental Atlas replay.",
        ],
    )


def export_replay(
    run, output, *, policy_path, configuration, atlas_root, python=sys.executable
):
    """Use the real Atlas converter; no adjacent source edits or compiled execution."""
    run, output, atlas_root = (
        Path(run).resolve(),
        Path(output).resolve(),
        Path(atlas_root).resolve(),
    )
    manifest, manifest_hash = _snapshot(run / "manifest.json")
    policy, policy_hash = _snapshot(policy_path)
    config, config_hash = _snapshot(configuration)
    for name, digest in manifest["artifacts"].items():
        if Path(name).name != name or sha256(run / name) != digest:
            raise ValueError(f"Invalid run artifact or hash: {name}")
    if sha256(manifest["experiment"]["shape_library"]) != manifest["shape_sha256"]:
        raise ValueError("Run shape library hash mismatch")
    if (
        "history.json" not in manifest["artifacts"]
        or "prepared-stage-0.h5" not in manifest["artifacts"]
    ):
        raise ValueError(
            "History and accepted-preparation checkpoint receipts required"
        )
    with h5py.File(run / "prepared-stage-0.h5") as f:
        spec = Experiment(**manifest["experiment"])
        if (
            f.attrs["format"] != spec.checkpoint_format
            or f.attrs["shape_sha256"] != manifest["shape_sha256"]
        ):
            raise ValueError("Prepared checkpoint identity mismatch")
        if spec.viscous:
            for name in ("normalDamping", "forceVelocity", "forceOmega"):
                if name not in f:
                    raise ValueError("Prepared checkpoint missing normal damping state")
                finite_array(f[name][:], (spec.particles, 4))
        prepared = {k: int(f.attrs[k]) for k in ["stage_index", "stage_steps", "step"]}
        prepared["initial_cell"] = json.loads(f.attrs["initial_cell"])
    history, history_hash = _snapshot(run / "history.json")
    if (
        history.get("format") != "sand-dem-history/1"
        or history.get("kind") != "synthetic-dem"
    ):
        raise ValueError("Unsupported source history")
    projected = project(manifest, history["rows"], prepared, policy)
    model = config["model"]
    if (
        not isinstance(model, str)
        or not model
        or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in model)
    ):
        raise ValueError("Invalid Atlas model identifier")
    consumer_files = [
        *sorted((atlas_root / "atlas").glob("*.py")),
        *sorted((atlas_root / "schemas").glob("*.json")),
        atlas_root / "models" / (model + ".json"),
        atlas_root / "web/recorded.mjs",
        atlas_root / "web/numerics.mjs",
        atlas_root / "web/recorded-view.mjs",
    ]
    if not (atlas_root / "atlas/experiment_replay.py").is_file():
        raise ValueError("Atlas checkout with prepare-replay support required")
    consumer_hashes = {
        str(p.relative_to(atlas_root)): sha256(p) for p in consumer_files
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "private").mkdir(mode=0o700)
    try:
        projected["source"] = dict(
            manifest_sha256=manifest_hash,
            history_sha256=history_hash,
            history_path=os.path.relpath(run / "history.json", output),
            shape_sha256=manifest["shape_sha256"],
            preparation_checkpoint_sha256=manifest["artifacts"]["prepared-stage-0.h5"],
            experiment=manifest["experiment"],
            source_hashes=manifest["source_hashes"],
            shader_hashes=manifest["shader_hashes"],
        )
        case_path = run.parent / "resolved-case.json"
        if case_path.exists():
            case = read_json(case_path)
            if (
                case["experiment"] != manifest["experiment"]
                or case["material"]["shape_sha256"] != manifest["shape_sha256"]
            ):
                raise ValueError("Resolved case disagrees with run")
            frozen = case.get("atlas_handoff")
            if frozen is not None and (
                policy != frozen["policy"] or config != frozen["configuration"]
            ):
                raise ValueError(
                    "Atlas policy/configuration differs from the pre-execution frozen case"
                )
            projected["source"]["case"] = case
            projected["source"]["resolved_case_sha256"] = sha256(case_path)
        write_json(output / "virtual-test.json", projected)
        # Legacy converter input is private and explicitly typed in provenance.
        # It is never sent to the measurement importer, whose labels are unsuitable.
        normalized = dict(
            schema_version="0.1",
            kind="experiment",
            id="synthetic-dem-" + manifest_hash[:16],
            material=projected["source"]
            .get("case", {})
            .get("material", {})
            .get("id", "shape-" + manifest["shape_sha256"][:16]),
            protocol="drained-triaxial",
            conventions={
                k: projected["conventions"][k]
                for k in ["stress", "strain", "stress_unit", "strain_unit"]
            },
            provenance=dict(
                source_url="https://github.com/SciGEM/sand-atlas-dem",
                evidence_kind="synthetic-dem",
                member_sha256=history_hash,
                archive_sha256=manifest_hash,
                projection_policy_sha256=policy_hash,
                virtual_test_sha256=sha256(output / "virtual-test.json"),
            ),
            initial=projected["initial"],
            rows=projected["rows"],
            warnings=projected["warnings"],
        )
        write_json(output / "private/replay-input.json", normalized)
        write_json(output / "configuration.json", config)
        write_json(output / "projection-policy.json", policy)
        command = [
            str(python),
            "-m",
            "atlas.cli",
            "prepare-replay",
            str(output / "private/replay-input.json"),
            str(output / "configuration.json"),
            "--output",
            str(output / "replay-request.json"),
        ]
        with (output / "converter.log").open("w") as log:
            subprocess.run(
                command,
                cwd=atlas_root,
                env={**os.environ, "PYTHONPATH": str(atlas_root)},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if any(
            sha256(atlas_root / name) != digest
            for name, digest in consumer_hashes.items()
        ):
            raise ValueError("Atlas consumer changed during conversion")
        if any(
            sha256(path) != digest
            for path, digest in [
                (run / "manifest.json", manifest_hash),
                (run / "history.json", history_hash),
                (policy_path, policy_hash),
                (configuration, config_hash),
            ]
        ):
            raise ValueError("Handoff inputs changed during conversion")
        if (output / "replay-request.json").stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Replay request exceeds the browser's 10 MiB import limit")
        report = dict(
            format="sand-dem-atlas-handoff/1",
            kind="synthetic-dem",
            status="complete",
            scientific_admission=False,
            consumer_source_hashes=consumer_hashes,
            configuration_sha256=config_hash,
            policy_sha256=policy_hash,
            source_manifest_sha256=manifest_hash,
            observation_count=len(projected["rows"]),
            interface="Atlas recorded-path request; no measurement overlay import",
            converter_command=command,
            artifacts={p.name: sha256(p) for p in output.iterdir() if p.is_file()},
        )
        write_json(output / "handoff.json", report)
        return report
    except Exception as exc:
        write_json(
            output / "handoff.json",
            dict(
                format="sand-dem-atlas-handoff/1",
                kind="synthetic-dem",
                status="failed",
                scientific_admission=False,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise
