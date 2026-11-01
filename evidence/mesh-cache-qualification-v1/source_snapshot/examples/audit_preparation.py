"""Independently audit a frozen periodic preparation, without executing the solver."""

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from sand_atlas_dem.util import read_json, sha256, write_json


def audit(study, policy):
    study = Path(study).resolve()
    frozen = read_json(study / "frozen-manifest.json")
    result = read_json(study / "run/result.json")
    if (
        frozen["status"] != "complete"
        or frozen["return_code"] != 0
        or result["status"] != "complete"
    ):
        raise ValueError("A terminal successful frozen study is required")
    for name, digest in frozen["sources"].items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or sha256(study / "source_snapshot" / relative) != digest
        ):
            raise ValueError("Frozen study source changed")
    if (
        policy["accepted_samples"] < 100
        or policy["pressure_relative_tolerance"] > 0.01
        or policy["maximum_kinetic_elastic_ratio"] > 1e-5
        or policy["maximum_unbalanced_force"] > 0.001
        or policy["maximum_overlap_fraction"] > 0.01
    ):
        raise ValueError(
            "Preparation policy cannot relax the established equilibrium gates"
        )
    latest = Path(result["latest_checkpoint"])
    if not latest.resolve().is_relative_to(study / "run"):
        raise ValueError("Study checkpoint is outside its frozen run")
    prepared = latest.parent / "prepared-stage-0.h5"
    inputs = {
        str(p): sha256(p)
        for p in [study / "frozen-manifest.json", study / "run/result.json", prepared]
    }
    requested = read_json(study / "source_snapshot/experiment-resolved.json")
    with h5py.File(prepared) as f:
        spec = json.loads(f.attrs["experiment"])
        if (
            f.attrs["format"] not in ("sand-dem-checkpoint/3", "sand-dem-checkpoint/5")
            or f.attrs["stage_index"] != 1
            or f.attrs["stage_steps"] != 0
            or f.attrs["step"] != result["step"]
        ):
            raise ValueError("Accepted prepared checkpoint identity required")
        if (
            len(spec["stages"]) != 1
            or spec["stages"][0]["kind"] != "isotropic"
            or spec.get("periodic", True) is not True
        ):
            raise ValueError("One-stage periodic isotropic preparation required")
        for name, value in requested.items():
            if name in ("contacts_per_particle", "pairs_per_particle", "bin_capacity"):
                if spec[name] < value:
                    raise ValueError("Checkpoint allocation shrank")
            elif spec.get(name) != value:
                raise ValueError(
                    "Checkpoint physical protocol differs from frozen experiment: "
                    + name
                )
        shape = Path(spec["shape_library"])
        inputs[str(shape)] = sha256(shape)
        if (
            inputs[str(shape)] != frozen["shape_sha256"]
            or f.attrs["shape_sha256"] != frozen["shape_sha256"]
        ):
            raise ValueError("Physical shape asset differs from study receipt")
        rows = json.loads(f.attrs["rows"])
        accepted_cell = np.array(json.loads(f.attrs["initial_cell"]))
        if (
            f["position"].shape != (spec["particles"], 4)
            or not np.isfinite(f["position"][:]).all()
        ):
            raise ValueError("Checkpoint particle inventory is invalid")
    if (
        spec["particles"] != policy["expected_particles"]
        or spec.get("contact_geometry", "sdf") != policy["expected_contact_geometry"]
    ):
        raise ValueError("Unexpected preparation inventory or geometry contract")
    if (
        not rows
        or any(row["stage"] != 0 for row in rows)
        or rows[-1]["step"] != result["step"]
        or any(
            b["step"] - a["step"] != spec["diagnostic_interval"]
            for a, b in zip(rows, rows[1:])
        )
        or not np.array_equal(accepted_cell, np.array(rows[-1]["cell"]))
    ):
        raise ValueError(
            "Complete ordered preparation history and accepted cell required"
        )
    count = policy["accepted_samples"]
    if len(rows) < count + 1:
        raise ValueError("Insufficient consecutive preparation observations")
    tail = rows[-count:]
    pressure = spec["stages"][0]["pressure"]
    stress = np.array([r["stress"] for r in tail], float)
    elastic = np.array([r["elastic_energy_proxy"] for r in tail], float)
    kinetic = np.array([r["kinetic_energy"] for r in tail], float)
    imbalance = np.array([r["unbalanced_force"] for r in tail], float)
    overlaps = np.array([r["max_overlap_fraction"] for r in rows], float)
    if (
        stress.shape != (count, 3, 3)
        or not all(
            np.isfinite(a).all()
            for a in (stress, elastic, kinetic, imbalance, overlaps)
        )
        or np.any(elastic <= 0)
        or np.any(kinetic < 0)
        or np.any(imbalance < 0)
        or np.any(overlaps < 0)
    ):
        raise ValueError("Finite physical preparation diagnostics required")
    pressure_error = float(
        np.max(abs(np.diagonal(stress, axis1=1, axis2=2) / pressure - 1))
    )
    energy_ratio = float(np.max(kinetic / elastic))
    with h5py.File(shape) as f:
        if f.attrs["format"] != "sand-dem-shapes/1" or f.attrs["units"] != "m":
            raise ValueError("Explicit SI shape metadata required")
        groups = [f[k] for k in sorted(f, key=int)]
        names = [str(g.attrs["id"]) for g in groups]
        volumes = np.array([g.attrs["volume"] for g in groups], float)
        radii = [g.attrs["radius"] for g in groups]
    if (
        len(names) != policy["expected_shapes"]
        or len(set(names)) != len(names)
        or not np.isfinite(volumes).all()
        or np.any(volumes <= 0)
    ):
        raise ValueError("Shape identity/volume inventory is invalid")
    # Checkpoint /3 and /5 reconstruct immutable IDs/scales from the frozen
    # constructor on resume. Execute only that own archived CPU placement function.
    source = study / "source_snapshot/sand_atlas_dem/simulation.py"
    tree = ast.parse(source.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "initial_assembly"
    )
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    metadata = SimpleNamespace(
        shapes=[SimpleNamespace(volume=v, radius=r) for v, r in zip(volumes, radii)]
    )
    _, _, stored_ids, stored_scales = namespace["initial_assembly"](
        metadata, SimpleNamespace(**spec)
    )
    if (
        stored_ids.shape != (spec["particles"],)
        or stored_ids.dtype.kind not in "iu"
        or np.any(stored_ids < 0)
        or np.any(stored_ids >= len(names))
        or stored_scales.shape != (spec["particles"],)
        or not np.allclose(stored_scales, spec["scale"], rtol=1e-7, atol=0)
    ):
        raise ValueError("Stored particle shape/scale inventory is invalid")
    counts = np.bincount(stored_ids.astype(np.int64), minlength=len(names))
    if spec.get("shape_sampling", "random") == "balanced":
        copies = policy["copies_per_shape"]
        if spec["particles"] != copies * len(names) or not np.all(counts == copies):
            raise ValueError("Stored balanced inventory differs from declared counts")
    elif len(names) != 1:
        raise ValueError(
            "This bounded auditor requires balanced sampling or one source shape"
        )
    solid_fraction = float(
        np.dot(volumes, counts) * spec["scale"] ** 3 / np.prod(accepted_cell)
    )
    if not np.isclose(
        1 / solid_fraction - 1, rows[-1]["void_ratio"], rtol=1e-10, atol=1e-12
    ):
        raise ValueError(
            "Independent rigid-volume void ratio disagrees with diagnostics"
        )
    pairs = min(r["active_grain_pairs"] for r in tail)
    passed = bool(
        pressure_error <= policy["pressure_relative_tolerance"]
        and energy_ratio <= policy["maximum_kinetic_elastic_ratio"]
        and max(imbalance) <= policy["maximum_unbalanced_force"]
        and max(overlaps) <= policy["maximum_overlap_fraction"]
        and solid_fraction >= policy["minimum_final_solid_fraction"]
        and pairs >= policy["minimum_active_grain_pairs"]
    )
    if any(sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("Preparation evidence changed during audit")
    return dict(
        format="sand-dem-preparation-audit/1",
        passed=passed,
        scientific_admission=False,
        prepared_checkpoint=str(prepared),
        prepared_checkpoint_sha256=inputs[str(prepared)],
        shape_library=str(shape),
        shape_library_sha256=inputs[str(shape)],
        step=result["step"],
        checked_samples=count,
        shape_counts=dict(zip(names, counts.tolist())),
        inventory_basis="Checkpoint resume contract: immutable IDs/scales reconstructed by archived initial_assembly; not mutable checkpoint fields.",
        max_pressure_error=pressure_error,
        max_energy_ratio=energy_ratio,
        max_force_imbalance=float(max(imbalance)),
        max_overlap=float(max(overlaps)),
        minimum_active_grain_pairs=int(pairs),
        final_solid_fraction=solid_fraction,
        policy=policy,
        input_hashes=inputs,
        source_unchanged=True,
        note="Independent recorded-state audit only. No solver execution, contact/timestep convergence, constitutive calibration or scientific admission is inferred.",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study", type=Path)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    policy = read_json(args.policy)
    inputs = {
        str(Path(__file__).resolve()): sha256(__file__),
        str(args.policy.resolve()): sha256(args.policy),
    }
    try:
        report = audit(args.study, policy)
        if any(sha256(path) != digest for path, digest in inputs.items()):
            raise ValueError("Auditor/policy changed")
        report["auditor_input_hashes"] = inputs
    except Exception as exc:
        report = dict(
            format="sand-dem-preparation-audit/1",
            passed=False,
            scientific_admission=False,
            error=f"{type(exc).__name__}: {exc}",
            auditor_input_hashes=inputs,
        )
    write_json(args.output, report)
    print("passed:", report["passed"], report.get("error", ""))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
