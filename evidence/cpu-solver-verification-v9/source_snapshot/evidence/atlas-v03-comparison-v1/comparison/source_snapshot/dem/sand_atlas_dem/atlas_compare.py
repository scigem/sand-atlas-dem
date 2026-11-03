"""Explicit synthetic DEM/continuum comparison through Atlas's native replay API."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .atlas_bridge import _snapshot
from .util import finite_array, sha256, write_json


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def comparison_model(catalogue, model_id):
    if catalogue.get("schema_version") != "0.1" or canonical_hash(
        catalogue["models"]
    ) != catalogue.get("sha256"):
        raise ValueError("Invalid generated Atlas catalogue identity")
    models = catalogue["models"]
    if len({m["id"] for m in models}) != len(models):
        raise ValueError("Duplicate compiled model identity")
    model = next((m for m in models if m["id"] == model_id), None)
    if model is None:
        raise ValueError("Replay model absent from compiled catalogue")
    version = model.get("schema_version")
    if version == "0.3":
        profile = model["formulation"]
        if (
            profile.get("time_dependence") != "strain_path"
            or profile.get("equation_form")
            not in ("incremental_residual", "strain_path_rate")
            or profile.get("kinematics") not in ("small_strain_3d", "axisymmetric_x")
            or profile.get("frame") != "fixed_spatial"
            or profile.get("objective_rate") != "none"
            or profile.get("tensor_representation") != "mandel"
            or profile.get("stress_sign") != "compression_positive"
            or profile.get("strain_sign") != "compression_positive"
            or profile.get("stress")
            != dict(source="state", name="sigma", measure="cauchy", scope="effective")
        ):
            raise ValueError(
                "Model formulation is unsupported by the incremental DEM replay projection"
            )
        binding = dict(profile["stress"])
    elif version == "0.1" and model.get("integration") in ("rate", "residual"):
        # Existing records retain the explicit legacy Atlas v0.1 consumer contract.
        # This creates no model record or migration and never admits physical time.
        binding = dict(
            source="state",
            name="sigma",
            measure="cauchy",
            scope="effective",
            contract="Atlas legacy model-v0.1 incremental contract",
        )
    else:
        raise ValueError("A supported incremental Atlas formulation is required")
    fields = model["state"]
    if (
        not fields
        or fields[0].get("name") != "sigma"
        or fields[0].get("shape") != "sym"
        or fields[0].get("unit") != "Pa"
    ):
        raise ValueError(
            "Declared stress binding must resolve to symmetric Pa state sigma"
        )
    if any(f.get("shape") not in ("sym", "scalar") for f in fields):
        raise ValueError("Unsupported compiled state layout")
    return model, binding


def comparison_rows(virtual, request, response, model, catalogue_hash):
    """Preserve every ordered observation, including reversals and duplicates."""
    if (
        virtual.get("format") != "sand-dem-atlas-virtual-test/1"
        or virtual.get("kind") != "synthetic-dem"
        or virtual.get("protocol") != "drained-triaxial"
        or response.get("ok") is not True
        or response.get("api_version") != "experimental-recorded-replay-0.1"
        or response.get("model") != request["model"]
        or response.get("catalogue_hash") != catalogue_hash
        or response.get("specification") != request
        or response.get("request_sha256") != canonical_hash(request)
        or response.get("parameters") != request["parameters"]
        or response.get("initial") != request["initial"]
        or response.get("initial_strain") != request["path"]["initial_strain"]
    ):
        raise ValueError("Native replay response does not match the synthetic handoff")
    knots, points, source = (
        request["path"]["knots"],
        response["points"],
        virtual["rows"],
    )
    if (
        not 1 <= len(source) <= 5001
        or len(knots) != len(points)
        or len(points) != len(source)
    ):
        raise ValueError("Complete original observation inventory required")
    if request["path"]["coordinate"] != dict(
        name="source_row_index", unit="1", meaning="path_order"
    ):
        raise ValueError("Synthetic handoff must retain source-row coordinates")
    size = sum(6 if f["shape"] == "sym" else 1 for f in model["state"])
    origin = finite_array(request["path"]["initial_strain"], (6,))
    void_offset, offset = None, 0
    for field in model["state"]:
        if field["name"] == "e" and field["shape"] == "scalar" and field["unit"] == "1":
            void_offset = offset
        offset += 6 if field["shape"] == "sym" else 1
    rows = []
    for index, (knot, point, observed) in enumerate(zip(knots, points, source)):
        if knot["coordinate"] != index or point["coordinate"] != index:
            raise ValueError("Replay changed original observation order")
        state = finite_array(point["state"], (size,))
        strain = finite_array(point["strain"], (6,))
        relative = strain - origin
        stress = state[:6]  # Resolved and validated binding, never an assumed output.
        finite_array(
            [
                observed[k]
                for k in (
                    "axial_strain",
                    "radial_strain",
                    "volumetric_strain",
                    "p",
                    "q",
                    "elapsed_time",
                )
            ]
        )
        continuum = dict(
            axial_strain=float(relative[0]),
            radial_strain=float(np.mean(relative[1:3])),
            volumetric_strain=float(sum(relative[:3])),
            p=float(np.mean(stress[:3])),
            q=float(stress[0] - np.mean(stress[1:3])),
            stress_mandel_Pa=stress.tolist(),
            strain_mandel=strain.tolist(),
            void_ratio=float(state[void_offset]) if void_offset is not None else None,
        )
        rows.append(
            dict(
                source_row_index=index,
                elapsed_time_s=observed["elapsed_time"],
                synthetic_dem=observed,
                continuum=continuum,
            )
        )
    return rows


_WORKER = """import json,sys
from atlas.model import load,physical_time
from atlas.recorded import execute_recorded
model=load(sys.argv[3])
if physical_time(model): raise ValueError("Recorded DEM replay requires incremental execution")
with open(sys.argv[1]) as f: request=json.load(f)
result=execute_recorded(request,sys.argv[2])
with open(sys.argv[4],"w") as f: json.dump(result,f,indent=2,allow_nan=False)
sys.exit(0 if result["ok"] else 1)
"""


def replay_comparison(
    handoff,
    output,
    *,
    atlas_root,
    solver,
    catalogue,
    plots=False,
    python=sys.executable,
):
    """Execute an existing native binary; never build or modify either atlas."""
    handoff, output, atlas_root, solver, catalogue = map(
        lambda p: Path(p).resolve(), (handoff, output, atlas_root, solver, catalogue)
    )
    receipt, receipt_hash = _snapshot(handoff / "handoff.json")
    if (
        receipt.get("format") != "sand-dem-atlas-handoff/1"
        or receipt.get("kind") != "synthetic-dem"
        or receipt.get("status") != "complete"
    ):
        raise ValueError("Complete synthetic DEM handoff required")
    inputs = {str(handoff / "handoff.json"): receipt_hash}
    for name, digest in receipt["artifacts"].items():
        if Path(name).name != name or sha256(handoff / name) != digest:
            raise ValueError("Handoff artifact identity mismatch: " + name)
        inputs[str(handoff / name)] = digest
    if (
        not {
            "replay-request.json",
            "virtual-test.json",
            "configuration.json",
            "projection-policy.json",
        }
        <= receipt["artifacts"].keys()
    ):
        raise ValueError("Handoff lacks required artifacts")
    request, _ = _snapshot(handoff / "replay-request.json")
    virtual, _ = _snapshot(handoff / "virtual-test.json")
    compiled, catalogue_hash = _snapshot(catalogue)
    model, binding = comparison_model(compiled, request["model"])
    if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in model["id"]):
        raise ValueError("Invalid compiled model identifier")
    model_path = atlas_root / "models" / (model["id"] + ".json")
    definition, definition_hash = _snapshot(model_path)
    if (
        definition != model
        or list(definition["parameters"]) != list(model["parameters"])
        or receipt["consumer_source_hashes"].get("models/" + model["id"] + ".json")
        != definition_hash
    ):
        raise ValueError("Model definition differs from handoff or compiled catalogue")
    if plots:
        import matplotlib

        matplotlib.use("Agg")
    sources = [
        *sorted((atlas_root / "atlas").glob("*.py")),
        *sorted((atlas_root / "schemas").glob("*.json")),
        model_path,
    ]
    own = Path(__file__).resolve().parent
    sources += [*sorted(own.glob("*.py")), *sorted((own / "shaders").glob("*.slang"))]
    inputs.update({str(p): sha256(p) for p in [solver, catalogue, *sources]})
    output.mkdir(parents=True, exist_ok=False)
    try:
        for path in sources:
            name = (
                "atlas/" + str(path.relative_to(atlas_root))
                if path.is_relative_to(atlas_root)
                else "dem/" + str(path.relative_to(own.parent))
            )
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        for name in (
            "replay-request.json",
            "virtual-test.json",
            "configuration.json",
            "projection-policy.json",
            "handoff.json",
        ):
            shutil.copy2(handoff / name, output / name)
        shutil.copy2(catalogue, output / "compiled-catalogue.json")
        write_json(
            output / "prospective-freeze.json",
            dict(
                inputs=inputs,
                stress_binding=binding,
                catalogue_sha256=compiled["sha256"],
                operation="Ordered native replay and synthetic comparison; no fitting or scientific admission",
            ),
        )
        described = subprocess.run(
            [str(solver), "--describe"], capture_output=True, timeout=60
        )
        (output / "native-describe.json").write_bytes(described.stdout)
        (output / "native-describe.log").write_bytes(described.stderr)
        actual, _ = _snapshot(output / "native-describe.json")
        if (
            described.returncode != 0
            or actual != compiled
            or [list(m["parameters"]) for m in actual["models"]]
            != [list(m["parameters"]) for m in compiled["models"]]
        ):
            raise ValueError("Native binary and supplied catalogue differ")
        with (output / "native.log").open("w") as log:
            result = subprocess.run(
                [
                    str(python),
                    "-c",
                    _WORKER,
                    str(output / "replay-request.json"),
                    str(solver),
                    str(model_path),
                    str(output / "native-response.json"),
                ],
                cwd=atlas_root,
                env={
                    **os.environ,
                    "PYTHONPATH": str(atlas_root),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=70,
            )
        if result.returncode != 0:
            raise RuntimeError(
                "Atlas replay failed; native response/log retain the failure prefix"
            )
        response, _ = _snapshot(output / "native-response.json")
        rows = comparison_rows(virtual, request, response, model, compiled["sha256"])
        differences = np.array(
            [
                [
                    row["continuum"][k] - row["synthetic_dem"][k]
                    for k in ("q", "p", "volumetric_strain")
                ]
                for row in rows
            ]
        )
        write_json(
            output / "comparison.json",
            dict(
                format="sand-dem-atlas-response-comparison/1",
                kind="synthetic-dem",
                scientific_admission=False,
                model=model["id"],
                stress_binding=binding,
                conventions=virtual["conventions"],
                rows=rows,
                maximum_absolute_difference=dict(
                    zip(
                        ("q_Pa", "p_Pa", "volumetric_strain"),
                        np.max(abs(differences), axis=0).tolist(),
                    )
                ),
                notes=[
                    "Continuum parameters are supplied, not fitted.",
                    "Radial stress is prescribed from the projected synthetic DEM history; response differences are not independent experimental validation.",
                    "Coordinates retain observation order; elapsed time is provenance, not physical-time integration.",
                    "Volumetric strain is the sum of engineering components, not finite-volume strain.",
                ],
            ),
        )
        if plots:
            _plot(rows, model["id"], output / "comparison.svg")
        changed = [path for path, digest in inputs.items() if sha256(path) != digest]
        if changed:
            raise ValueError(
                "Comparison input/source changed during execution: "
                + ", ".join(changed)
            )
        report = dict(
            format="sand-dem-atlas-response-report/1",
            kind="synthetic-dem",
            status="complete",
            scientific_admission=False,
            model=model["id"],
            observation_count=len(rows),
            binary_sha256=inputs[str(solver)],
            catalogue_sha256=compiled["sha256"],
            source_unchanged=True,
            artifacts={p.name: sha256(p) for p in output.iterdir() if p.is_file()},
        )
        write_json(output / "report.json", report)
        return report
    except Exception as exc:
        write_json(
            output / "report.json",
            dict(
                format="sand-dem-atlas-response-report/1",
                kind="synthetic-dem",
                status="failed",
                scientific_admission=False,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise


def _plot(rows, model, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for key, label, style in (
        ("synthetic_dem", "Synthetic source", "o-"),
        ("continuum", "Atlas: " + model, "--"),
    ):
        x = [row[key]["axial_strain"] * 100 for row in rows]
        axes[0].plot(x, [row[key]["q"] / 1000 for row in rows], style, label=label)
        axes[1].plot(
            x, [row[key]["volumetric_strain"] * 100 for row in rows], style, label=label
        )
    for axis in axes:
        axis.set_xlabel("Axial compression strain (%)")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Signed axial − radial stress (kPa)")
    axes[1].set_ylabel("Sum of engineering compression strains (%)")
    figure.suptitle("Synthetic comparison • no calibration or experimental validation")
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(figure)
