"""Named, hash-pinned materials and reproducible case orchestration (SI units)."""

import re
from pathlib import Path

from .spec import Experiment
from .util import read_json, sha256, write_json


class MaterialNotFound(ValueError):
    """No registered identity; distinct from ambiguous names or corrupt assets."""


def _material(record):
    required = {
        "format",
        "id",
        "aliases",
        "shape_library",
        "shape_sha256",
        "source",
        "scale_evidence",
    }
    if set(record) != required or record["format"] != "sand-dem-material/1":
        raise ValueError(
            "Expected sand-dem-material/1 with explicit source and scale_evidence"
        )
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", record["id"]):
        raise ValueError("Material id must be a lowercase canonical identifier")
    if not isinstance(record["aliases"], list) or any(
        not isinstance(a, str) or not a.strip() for a in record["aliases"]
    ):
        raise ValueError("Material aliases must be nonempty strings")
    if not re.fullmatch(r"[a-f0-9]{64}", record["shape_sha256"]):
        raise ValueError("Invalid shape_sha256")
    if not isinstance(record["source"], dict) or not record["source"]:
        raise ValueError("Explicit source receipt required")
    if (
        not isinstance(record["scale_evidence"], str)
        or not record["scale_evidence"].strip()
    ):
        raise ValueError("Source-supported coordinate scale evidence required")
    return record


def materials(cache):
    """List registered receipts; resolution separately verifies the asset bytes."""
    return [_material(read_json(p)) for p in sorted(Path(cache).glob("*.json"))]


def register_material(receipt, cache):
    receipt = Path(receipt).resolve()
    record = _material(read_json(receipt))
    library = (receipt.parent / record["shape_library"]).resolve()
    if sha256(library) != record["shape_sha256"]:
        raise ValueError("Material shape library hash mismatch")
    from .geometry import ShapeLibrary

    ShapeLibrary.load(
        library
    )  # Validate the prepared representation before registration.
    record["shape_library"] = str(library)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{record['id']}.json"
    with target.open("x") as f:
        import json

        json.dump(record, f, indent=2, allow_nan=False)
        f.write("\n")
    return record


def resolve_material(name, cache):
    key = name.strip().casefold()
    records = materials(cache)
    exact = [r for r in records if r["id"] == key]
    matches = exact or [
        r for r in records if key in [a.strip().casefold() for a in r["aliases"]]
    ]
    if not matches:
        raise MaterialNotFound(
            f"Material {name!r} is not registered; use material prepare/register"
        )
    if len(matches) != 1:
        raise ValueError(
            f"Material {name!r}: expected one match, found {len(matches)}; use material list/register"
        )
    record = matches[0]
    if sha256(record["shape_library"]) != record["shape_sha256"]:
        raise ValueError(
            "Material shape library hash mismatch; re-prepare or restore the registered asset"
        )
    return record


def resolve_case(material, case, cache, overrides=None):
    case = Path(case)
    raw = read_json(case)
    if set(raw) != {"format", "experiment"} or raw["format"] != "sand-dem-case/1":
        raise ValueError("Expected sand-dem-case/1 with an experiment object")
    values = dict(raw["experiment"])
    if {"format", "shape_library"} & values.keys():
        raise ValueError("Case geometry is supplied by the named material")
    required = {
        "density",
        "normal_stiffness",
        "tangential_stiffness",
        "friction",
        "timestep",
        "stages",
    }
    if not required <= values.keys():
        raise ValueError(
            f"Explicit case parameters required: {sorted(required - values.keys())}"
        )
    # Copy stage dictionaries so CLI overrides cannot mutate the input record.
    values["stages"] = [dict(s) for s in values["stages"]]
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    if set(overrides) - {"particles", "seed", "pressure", "rate", "stop_strain"}:
        raise ValueError("Unknown case override")
    for key, value in overrides.items():
        if key in {"particles", "seed"}:
            values[key] = value
        else:
            kinds = {"isotropic", "triaxial"} if key == "pressure" else {"triaxial"}
            targets = [s for s in values["stages"] if s["kind"] in kinds]
            if not targets:
                raise ValueError(f"Override {key} has no applicable stage")
            for stage in targets:
                stage[key] = value
    Experiment(**values)  # Validate all case fields before any source acquisition.
    record = resolve_material(material, cache)
    spec = Experiment(shape_library=record["shape_library"], **values)
    steps = sum(s["steps"] for s in spec.stages)
    return dict(
        format="sand-dem-resolved-case/1",
        kind="synthetic-dem",
        scientific_admission=False,
        case_sha256=sha256(case),
        material=record,
        overrides=overrides,
        experiment=spec.to_dict(),
        budget=dict(
            maximum_steps=steps,
            maximum_physical_time_s=steps * spec.timestep,
            shape_library_bytes=Path(record["shape_library"]).stat().st_size,
            working_memory_limit_gib=spec.working_memory_gib,
            wall_time_estimate_s=None,
        ),
        notes=[
            "Material names supply geometry, not calibrated contact parameters.",
            "Step budgets are upper bounds; no case-specific throughput estimate is available.",
            "The current comparison bundle requires a synthetic-test adapter in Constitutive Atlas.",
        ],
    )


def run_case(resolved, output, max_chunks=None):
    """Keep resolved inputs even if device initialization or preparation fails."""
    from .simulation import Simulation, export_run

    if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 1):
        raise ValueError("max_chunks must be a positive integer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "resolved-case.json", resolved)
    try:
        spec = Experiment(**resolved["experiment"])
        if sha256(spec.shape_library) != resolved["material"]["shape_sha256"]:
            raise ValueError("Material changed after case resolution")
        result = Simulation(spec).run(output / "run", max_chunks=max_chunks)
        if result.rows:
            export_run(result.path, output / "exports")
            bundle_path = output / "exports" / "atlas-comparison.json"
            bundle = read_json(bundle_path)
            bundle["case_provenance"] = dict(
                material=resolved["material"],
                case_sha256=resolved["case_sha256"],
                resolved_case_sha256=sha256(output / "resolved-case.json"),
            )
            write_json(bundle_path, bundle)
        report = dict(
            format="sand-dem-case-result/1",
            kind="synthetic-dem",
            status=result.status,
            scientific_admission=False,
            resolved_case_sha256=sha256(output / "resolved-case.json"),
            rows=len(result.rows),
            path=str(output),
        )
        write_json(output / "result.json", report)
        return report
    except Exception as exc:
        write_json(
            output / "result.json",
            dict(
                format="sand-dem-case-result/1",
                kind="synthetic-dem",
                status="failed",
                scientific_admission=False,
                error=f"{type(exc).__name__}: {exc}",
                resolved_case_sha256=sha256(output / "resolved-case.json"),
            ),
        )
        raise
