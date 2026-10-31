"""Exercise reference -> mixed asset -> actual Metal comparison on analytical boxes.

Own synthetic geometry and independent exact box distance/continuum integrals.
This verifies the research workflow, not Hostun or assembly material behaviour.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.geometry import ShapeLibrary, from_mesh, with_face_quadrature
from sand_atlas_dem.quadrature import prepare_adaptive
from sand_atlas_dem.util import read_json, sha256, write_json


def subdivide(triangles):
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    return np.stack(
        [
            np.stack([a, ab, ca], 1),
            np.stack([ab, b, bc], 1),
            np.stack([ca, bc, c], 1),
            np.stack([ab, bc, ca], 1),
        ],
        1,
    ).reshape(-1, 3, 3)


def box_force(points, weights, extents, separation):
    """Exact interior nearest-face distance, independent of triangle queries."""
    total, active, maximum_depth = np.zeros(3), 0, 0.0
    for sign in (1, -1):
        q = points - [sign * separation, 0, 0]
        clearance = extents / 2 - np.abs(q)
        inside = np.all(clearance > 0, axis=1)
        q, gaps, w = q[inside], clearance[inside], weights[inside]
        axis = np.argmin(gaps, axis=1)
        distance = gaps[np.arange(len(gaps)), axis]
        normals = np.zeros_like(q)
        normals[np.arange(len(q)), axis] = np.sign(q[np.arange(len(q)), axis])
        total += sign * 0.5 * np.sum((w * distance)[:, None] * normals, axis=0)
        active += len(q)
        maximum_depth = max(maximum_depth, float(distance.max(initial=0)))
    return total, active, maximum_depth


def references(shapes, policy):
    rows = []
    bary = np.full((3, 3), 1 / 6) + np.eye(3) / 2
    for shape, extents in zip(shapes, policy["box_extents_m"]):
        extents = np.array(extents)
        diameter = (6 * shape.volume / np.pi) ** (1 / 3)
        cases = [
            dict(
                shape=shape.id,
                target_sampled_depth_fraction=fraction,
                separation_m=extents[0] - fraction * diameter,
                levels=[],
            )
            for fraction in policy["depth_fractions"]
        ]
        triangles = shape.vertices[shape.faces]
        for level in range(7):
            points = np.einsum("ij,fjk->fik", bary, triangles).reshape(-1, 3)
            area = (
                np.linalg.norm(
                    np.cross(
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0],
                    ),
                    axis=1,
                )
                / 2
            )
            weights = np.repeat(area / 3, 3)
            for case in cases:
                force, active, depth = box_force(
                    points, weights, extents, case["separation_m"]
                )
                case["levels"].append(
                    dict(
                        level=level,
                        force_integral_m3=force,
                        active_nodes=active,
                        maximum_sampled_depth_fraction=depth / diameter,
                    )
                )
            if level < 6:
                triangles = subdivide(triangles)
        for case in cases:
            force, previous = (
                np.array(case["levels"][i]["force_integral_m3"]) for i in (-1, -2)
            )
            depth = extents[0] - case["separation_m"]
            # Only the central face square/rectangle has nearest normal along x.
            exact = np.array(
                [-depth * (extents[1] - 2 * depth) * (extents[2] - 2 * depth), 0, 0]
            )
            assert depth > 0 and 2 * depth < extents.min()
            case.update(
                reference_finest_level=6,
                finest_relative_change=float(
                    np.linalg.norm(force - previous) / np.linalg.norm(force)
                ),
                closed_continuum_integral_m3=exact,
                relative_continuum_error=float(
                    np.linalg.norm(force - exact) / np.linalg.norm(exact)
                ),
            )
        rows.extend(cases)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = Path(__file__).resolve().parents[1], args.output.resolve()
    policy = read_json(args.protocol)
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
    write_json(output / "prospective-freeze.json", dict(files=hashes))
    write_json(output / "protocol.json", policy)
    shapes = [
        from_mesh(
            trimesh.creation.box(extents=extent),
            shape_id=f"analytical-box-{i}",
            unit="m",
            resolution=8,
            samples=32,
        )
        for i, extent in enumerate(policy["box_extents_m"])
    ]
    parent = output / "private/parent.h5"
    ShapeLibrary(shapes).save(parent)
    reference_dir = output / "reference"
    reference_dir.mkdir()
    protocol = dict(
        identity=policy["identity"] + "-reference",
        library_sha256=sha256(parent),
        shape_ids=[s.id for s in shapes],
        target_sampled_depth_fractions=policy["depth_fractions"],
        nearest_point_policy="exact-analytical-axis-aligned-box",
        inside_policy="strict-positive-face-clearance",
    )
    write_json(reference_dir / "protocol.json", protocol)
    # The accepted reference archive binds every used source, before evaluation.
    shutil.copytree(output / "source_snapshot", reference_dir / "source_snapshot")
    write_json(
        reference_dir / "prospective-freeze.json",
        dict(
            source_hashes=hashes,
            protocol_sha256=sha256(reference_dir / "protocol.json"),
        ),
    )
    rows = references(shapes, policy)
    reference_passed = all(
        r["finest_relative_change"] <= policy["maximum_reference_change"]
        and r["relative_continuum_error"] <= policy["maximum_reference_continuum_error"]
        for r in rows
    )
    write_json(
        reference_dir / "report.json",
        dict(
            passed=reference_passed,
            source_unchanged=True,
            reference_refinement_passed=reference_passed,
            expected_cases=len(rows),
            completed_cases=len(rows),
            failures=[],
            rows=rows,
            scope="Own analytical box distance and closed continuum force",
        ),
    )
    if not reference_passed:
        raise RuntimeError(
            "Analytical reference failed its frozen continuum/refinement gates"
        )
    candidate = output / "private/adaptive.h5"
    receipt = prepare_adaptive(
        parent,
        reference_dir / "report.json",
        reference_dir / "protocol.json",
        candidate,
        target=policy["selection_target"],
    )
    write_json(output / "preparation-receipt.json", receipt)
    levels = {r["shape"]: r["subdivisions"] for r in receipt["shapes"]}
    if len(set(levels.values())) < policy["minimum_distinct_selected_levels"]:
        raise RuntimeError("Fixture did not exercise distinct per-shape resolutions")
    coarse = output / "private/coarse.h5"
    coarse_shapes = [with_face_quadrature(s, "gauss3", 0) for s in shapes]
    for shape in coarse_shapes:
        shape.provenance["parent_shape_library_sha256"] = sha256(parent)
    ShapeLibrary(coarse_shapes).save(coarse)
    gpu_policy = dict(
        identity=policy["identity"] + "-metal",
        reference_report=str(reference_dir / "report.json"),
        reference_report_sha256=sha256(reference_dir / "report.json"),
        reference_protocol=str(reference_dir / "protocol.json"),
        reference_protocol_sha256=sha256(reference_dir / "protocol.json"),
        reference_freeze_sha256=sha256(reference_dir / "prospective-freeze.json"),
        parent_library_sha256=sha256(parent),
        libraries=[
            dict(path=str(coarse), sha256=sha256(coarse), subdivisions=0),
            dict(
                path=str(candidate),
                sha256=sha256(candidate),
                subdivisions_by_shape=levels,
            ),
        ],
        maximum_relative_force_error=policy["maximum_gpu_integral_error"],
        maximum_fixed_level_gpu_error=policy["maximum_gpu_fixed_level_error"],
        minimum_worst_error_improvement_factor=policy["minimum_gpu_refinement_factor"],
        timestep_s=policy["timestep_s"],
        overlap_guard_fraction=policy["overlap_guard_fraction"],
        contacts_per_particle=4096,
        zero_fixed_reference="require-exact-zero-force-and-zero-active-nodes",
        scope="Analytical workflow fixture only; Hostun fixed-node baseline and full-library qualification remain separate",
    )
    gpu_policy_path = output / "gpu-protocol.json"
    write_json(gpu_policy_path, gpu_policy)
    with (output / "gpu.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(root / "examples/mesh_quadrature_study.py"),
                str(gpu_policy_path),
                "--output",
                str(output / "gpu"),
            ],
            env={**os.environ, "PYTHONPATH": str(root / "src")},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    gpu = (
        read_json(output / "gpu/report.json")
        if (output / "gpu/report.json").exists()
        else {}
    )
    unchanged = all(sha256(root / name) == digest for name, digest in hashes.items())
    passed = bool(result.returncode == 0 and gpu.get("passed") and unchanged)
    write_json(
        output / "report.json",
        dict(
            identity=policy["identity"],
            passed=passed,
            source_unchanged=unchanged,
            selected_levels=levels,
            reference_cases=len(rows),
            gpu_cases=len(gpu.get("rows", [])),
            reference_maximum_continuum_error=max(
                r["relative_continuum_error"] for r in rows
            ),
            gpu_coarse_worst_relative_error=gpu.get("coarse_worst_relative_error"),
            gpu_fine_worst_relative_error=gpu.get("fine_worst_relative_error"),
            backend="metal",
            scientific_admission=False,
            scope=__doc__,
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
