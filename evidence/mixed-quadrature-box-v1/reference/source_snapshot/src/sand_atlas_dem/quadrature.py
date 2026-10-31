"""Reference-driven fixed surface quadrature, selected before GPU qualification.

This changes preprocessing resolution per shape, never during a simulation.
Reference-pose agreement is not dynamic or material qualification.
"""

from pathlib import Path

import numpy as np

from .geometry import ShapeLibrary, with_face_quadrature
from .util import finite_array, read_json, sha256


def select_levels(reference, protocol, shape_ids, *, target=0.05):
    """Choose the smallest level >=2 satisfying every supplied pose of a shape.

    A selected level must be strictly coarser than its independent reference.
    The existing 2% finest-reference-change gate remains mandatory.
    """
    if not np.isfinite(target) or not 0 < target <= 0.1:
        raise ValueError("Quadrature selection target must be in (0, 0.1]")
    if (
        not all(
            reference.get(name) is True
            for name in ("passed", "source_unchanged", "reference_refinement_passed")
        )
        or reference.get("failures") != []
    ):
        raise ValueError("A passing source-stable reference study is required")
    shape_ids = list(shape_ids)
    if (
        len(set(shape_ids)) != len(shape_ids)
        or sorted(shape_ids) != protocol["shape_ids"]
    ):
        raise ValueError("Source shape inventory does not match reference protocol")
    depths = protocol["target_sampled_depth_fractions"]
    if len(set(depths)) != len(depths) or not depths:
        raise ValueError("Distinct reference depth identities required")
    expected = {(name, depth) for name in shape_ids for depth in depths}
    rows = reference["rows"]
    keys = [(r["shape"], r["target_sampled_depth_fraction"]) for r in rows]
    if (
        len(keys) != len(expected)
        or set(keys) != expected
        or reference.get("completed_cases") != len(expected)
        or reference.get("expected_cases") != len(expected)
    ):
        raise ValueError("Complete unique reference pose inventory required")
    by_shape = {name: [] for name in shape_ids}
    for row in rows:
        levels = row["levels"]
        finest = row["reference_finest_level"]
        if finest not in (4, 6) or [v["level"] for v in levels] != list(
            range(finest + 1)
        ):
            raise ValueError("Complete ordered reference levels required")
        forces = np.array([finite_array(v["force_integral_m3"], (3,)) for v in levels])
        norm = np.linalg.norm(forces[-1])
        if norm <= 0:
            raise ValueError("Nonzero independent reference force required")
        errors = np.linalg.norm(forces - forces[-1], axis=1) / norm
        if errors[-2] > 0.02 or not np.isclose(
            errors[-2], row["finest_relative_change"], rtol=1e-10, atol=1e-14
        ):
            raise ValueError("Reference failed its unchanged 2% refinement gate")
        by_shape[row["shape"]].append(
            dict(
                target_sampled_depth_fraction=row["target_sampled_depth_fraction"],
                separation_m=row["separation_m"],
                reference_finest_level=finest,
                level_relative_errors=errors.tolist(),
            )
        )
    selection = []
    for name in sorted(by_shape):
        cases = by_shape[name]
        candidates = [
            level
            for level in range(2, 5)
            if all(
                level < case["reference_finest_level"]
                and case["level_relative_errors"][level] <= target
                for case in cases
            )
        ]
        if not candidates:
            raise ValueError(f"No independently supported quadrature level for {name}")
        level = candidates[0]
        selection.append(
            dict(
                shape=name,
                subdivisions=level,
                worst_reference_relative_error=max(
                    c["level_relative_errors"][level] for c in cases
                ),
                poses=sorted(cases, key=lambda c: c["target_sampled_depth_fraction"]),
            )
        )
    return selection


def prepare_adaptive(
    library_path, reference_path, protocol_path, output, *, target=0.05
):
    """Create a separate asset and receipt; reject failed/incomplete references."""
    library_path, reference_path, protocol_path, output = map(
        Path, (library_path, reference_path, protocol_path, output)
    )
    if output.exists():
        raise FileExistsError(output)
    freeze_path = reference_path.parent / "prospective-freeze.json"
    inputs = {
        str(path.resolve()): sha256(path)
        for path in (library_path, reference_path, protocol_path, freeze_path)
    }
    parent_hash, reference_hash, protocol_hash = (
        inputs[str(p.resolve())] for p in (library_path, reference_path, protocol_path)
    )
    protocol = read_json(protocol_path)
    freeze = read_json(freeze_path)
    if freeze.get("protocol_sha256") != protocol_hash or not freeze.get(
        "source_hashes"
    ):
        raise ValueError("Reference protocol does not match its prospective freeze")
    for name, digest in freeze["source_hashes"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid archived source path")
        if sha256(reference_path.parent / "source_snapshot" / relative) != digest:
            raise ValueError("Reference source snapshot differs from its freeze")
    if protocol["library_sha256"] != parent_hash:
        raise ValueError("Physical library hash differs from reference protocol")
    reference = read_json(reference_path)
    library = ShapeLibrary.load(library_path)
    selection = select_levels(
        reference, protocol, [s.id for s in library.shapes], target=target
    )
    chosen = {r["shape"]: r for r in selection}
    shapes = []
    for shape in library.shapes:
        record = chosen[shape.id]
        new = with_face_quadrature(shape, "gauss3", record["subdivisions"])
        new.provenance.update(
            parent_shape_library_sha256=parent_hash,
            quadrature_reference_report_sha256=reference_hash,
            quadrature_reference_protocol_sha256=protocol_hash,
            quadrature_selection_target_relative_error=target,
            quadrature_selection_worst_relative_error=record[
                "worst_reference_relative_error"
            ],
            quadrature_gpu_verified=False,
            material_qualified=False,
        )
        record["sample_count"] = len(new.samples)
        record["source_face_count"] = len(shape.faces)
        shapes.append(new)
    if any(sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("Quadrature preparation input changed during execution")
    ShapeLibrary(shapes).save(output)
    return dict(
        format="sand-dem-adaptive-quadrature/1",
        output=str(output.resolve()),
        sha256=sha256(output),
        parent_sha256=parent_hash,
        reference_report_sha256=reference_hash,
        reference_protocol_sha256=protocol_hash,
        reference_freeze_sha256=inputs[str(freeze_path.resolve())],
        rule="gauss3",
        selection_target_relative_error=target,
        minimum_level=2,
        maximum_level=4,
        shapes=selection,
        total_samples=sum(len(s.samples) for s in shapes),
        static_reference_target_passed=True,
        gpu_verified=False,
        scientific_admission=False,
        note="Fixed per-shape preprocessing; requires separate matching GPU and dynamic assembly qualification. New sample keys cannot replace a continuing contact history.",
    )
