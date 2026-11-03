"""Compare completed frozen mesh workloads without conflating timing and accuracy."""

import argparse
from pathlib import Path

from sand_atlas_dem.util import read_json, sha256, write_json


def compare(before, after):
    roots = [Path(before), Path(after)]
    manifests = [read_json(p / "frozen-manifest.json") for p in roots]
    if any(m["status"] != "complete" for m in manifests):
        raise ValueError("Both frozen benchmark runs must be complete")
    protocols = [read_json(p / "source_snapshot/protocol-original.json") for p in roots]
    for key in (
        "particles",
        "orientation_seed",
        "packing",
        "library_sha256",
        "target_maximum_penetration_fraction_of_equivalent_diameter",
        "reset_to_identical_committed_state_before_each_repeat",
        "reset_and_diagnostic_time_in_per_step_timing",
    ):
        if protocols[0][key] != protocols[1][key]:
            raise ValueError(f"Frozen protocol differs: {key}")
    for root, manifest in zip(roots, manifests):
        for name, expected in manifest["sources"].items():
            if sha256(root / "source_snapshot" / name) != expected:
                raise ValueError(f"Frozen source changed: {root}, {name}")
        if sha256(root / "private/shapes.h5") != manifest["shape_sha256"]:
            raise ValueError(f"Frozen shape library changed: {root}")
    rows = []
    for count in protocols[0]["particles"]:
        paths = [p / f"particles-{count}/benchmark.json" for p in roots]
        reports = [read_json(p) for p in paths]
        for key in (
            "particles",
            "unique_shapes",
            "samples",
            "packing",
            "contact_geometry",
            "controlled_pair_separations_m",
            "timestep_s",
            "pairs_per_particle",
            "contacts_per_particle",
            "minimum_wall_seconds",
            "reset_time_included",
            "shape_library_sha256",
            "steps_per_repeat",
            "requested_repeats",
        ):
            if reports[0]["workload"][key] != reports[1]["workload"][key]:
                raise ValueError(f"Workload differs: {count}, {key}")
        for report in reports:
            if report["workload"]["contact_geometry"] != "mesh":
                raise ValueError("Expected original-triangle mesh workloads")
        for phase in ("initial_diagnostics", "final_diagnostics"):
            for key in ("active_grain_pairs", "active_surface_contacts"):
                if reports[0][phase][key] != reports[1][phase][key]:
                    raise ValueError(f"Contact counts differ: {count}, {phase}, {key}")
        old, new = [r["median_seconds_per_step"] for r in reports]
        rows.append(
            {
                "particles": count,
                "report_sha256": [sha256(p) for p in paths],
                "before_seconds_per_step": old,
                "after_seconds_per_step": new,
                "speedup": old / new,
                "after_hours_per_million_steps": new * 1e6 / 3600,
                "physical_seconds_per_million_steps": reports[1]["workload"][
                    "timestep_s"
                ]
                * 1e6,
                "after_logical_solver_buffer_bytes": reports[1]["memory_accounting"][
                    "logical_solver_buffer_bytes"
                ],
                "active_surface_contacts": reports[1]["final_diagnostics"][
                    "active_surface_contacts"
                ],
                "after_first_last_quarter_seconds_per_step": [
                    reports[1]["first_quarter_median_seconds_per_step"],
                    reports[1]["last_quarter_median_seconds_per_step"],
                ],
            }
        )
    return {
        "format": "sand-dem-mesh-performance-comparison/1",
        "before": str(roots[0]),
        "after": str(roots[1]),
        "rows": rows,
        "scope": "Matched isolated-pair workloads and contact counts. Wall time excludes reset/diagnostics. No dense-specimen, quadrature convergence or experimental validation claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, compare(args.before, args.after))


if __name__ == "__main__":
    main()
