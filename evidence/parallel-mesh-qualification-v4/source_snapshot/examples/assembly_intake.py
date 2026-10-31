"""CPU geometry/resource intake only; this never prepares or simulates a specimen."""

import argparse
import math
import shutil
from pathlib import Path

import numpy as np

from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.simulation import initial_assembly
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    policy = read_json(args.policy)
    protocol = Path(policy["experiment"])
    spec = Experiment.load(protocol)
    parent = Path(policy["intake_parent_library"])
    if sha256(parent) != policy["intake_parent_library_sha256"]:
        raise ValueError("Frozen physical geometry parent changed")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__),
        args.policy.resolve(),
        protocol.resolve(),
        *sorted((root / "src/sand_atlas_dem").glob("*.py")),
        *sorted((root / "src/sand_atlas_dem/shaders").glob("*.slang")),
    ]
    hashes = {str(p.relative_to(root)): sha256(p) for p in paths}
    for path in paths:
        target = output / "source_snapshot" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(
        output / "prospective-freeze.json",
        dict(
            files=hashes,
            physical_parent_sha256=policy["intake_parent_library_sha256"],
            scope=__doc__,
            acceptance="Exact balanced inventory, positive independent periodic bounding-sphere gap, declared timestep below algebraic bound, periodic image constraint permits nominal solid fraction0.65; unchanged inputs. No GPU or preparation acceptance.",
        ),
    )
    library = ShapeLibrary.load(parent)
    positions, cell, ids, scales = initial_assembly(library, spec)
    radius = np.array([library.shapes[int(i)].radius * s for i, s in zip(ids, scales)])
    branch = positions[:, None, :] - positions[None, :, :]
    branch -= cell * np.rint(branch / cell)
    gaps = np.linalg.norm(branch, axis=2) - radius[:, None] - radius[None, :]
    np.fill_diagonal(gaps, np.inf)
    counts = np.bincount(ids, minlength=len(library.shapes))
    volume = sum(library.shapes[int(i)].volume * s**3 for i, s in zip(ids, scales))
    cutoff = 2 * max(radius) * (1 + spec.skin)
    image_limit = volume / (2 * cutoff) ** 3
    limits = []
    if spec.normal_viscosity or spec.planes:
        raise ValueError(
            "This declared intake estimates undamped periodic grain contacts only"
        )
    for shape in library.shapes:
        stiffness = (
            (spec.normal_stiffness + spec.tangential_stiffness)
            * shape.areas.sum()
            * spec.scale**2
        )
        mass = spec.density * shape.volume * spec.scale**3
        inertia = np.linalg.eigvalsh(spec.density * shape.inertia * spec.scale**5).min()
        limits.append(
            0.1
            * math.sqrt(
                min(
                    mass / stiffness,
                    inertia / (stiffness * (shape.radius * spec.scale) ** 2),
                )
            )
        )
    unchanged = (
        all(sha256(root / name) == digest for name, digest in hashes.items())
        and sha256(parent) == policy["intake_parent_library_sha256"]
    )
    inventory = (
        len(library.shapes) == policy["expected_shapes"]
        and spec.particles == policy["expected_shapes"] * policy["copies_per_shape"]
        and np.all(counts == policy["copies_per_shape"])
    )
    passed = bool(
        inventory
        and gaps.min() > 0
        and spec.timestep <= min(limits)
        and image_limit > 0.65
        and unchanged
    )
    report = dict(
        identity=policy["identity"] + "-intake",
        passed=passed,
        source_unchanged=unchanged,
        particles=spec.particles,
        shape_counts={s.id: int(n) for s, n in zip(library.shapes, counts)},
        cell_m=cell.tolist(),
        initial_solid_fraction=volume / float(np.prod(cell)),
        minimum_independent_periodic_bounding_gap_m=float(gaps.min()),
        conservative_parent_timestep_bound_s=min(limits),
        declared_timestep_s=spec.timestep,
        periodic_image_limit_solid_fraction=image_limit,
        maximum_steps=spec.stages[0]["steps"],
        maximum_physical_time_s=spec.stages[0]["steps"] * spec.timestep,
        controller_update_period_s=spec.diagnostic_interval * spec.timestep,
        acceptance_window_s=policy["accepted_samples"]
        * spec.diagnostic_interval
        * spec.timestep,
        gpu_executed=False,
        preparation_accepted=False,
        scientific_admission=False,
        note="Physical parent geometry is used only for placement and algebraic preflight. Contact quadrature/GPU qualification and actual adaptive-asset device preflight remain mandatory. The periodic image ceiling is a computational bound, not an achievable packing density or specimen acceptance.",
    )
    write_json(output / "report.json", report)
    print(
        "passed:",
        passed,
        "minimum gap m:",
        gaps.min(),
        "timestep bound s:",
        min(limits),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
