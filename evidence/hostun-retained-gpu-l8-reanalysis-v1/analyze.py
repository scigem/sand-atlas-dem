"""Reanalyze immutable observed GPU forces with stronger same-pose references.

No new GPU execution and no numerical gate revisions. Retain the original study.
"""

import ast
from pathlib import Path

import numpy as np

from sand_atlas_dem.util import read_json, sha256, write_json

out = Path(__file__).resolve().parent
root = out.parents[1]
gpu_path = root / "evidence/hostun-library-gpu-quadrature-adaptive-v3/gpu/report.json"
ref_path = root / "evidence/hostun-library-quadrature-consolidated-v4/report.json"
gpu, reference = read_json(gpu_path), read_json(ref_path)
policy = gpu["protocol"]
baseline_path = Path(policy["fixed_node_baseline"]["path"])
inputs = {
    str(p): sha256(p) for p in (gpu_path, ref_path, baseline_path, Path(__file__))
}
write_json(
    out / "prospective-freeze.json",
    dict(
        inputs=inputs,
        maximum_relative_force_error=policy["maximum_relative_force_error"],
        maximum_fixed_level_gpu_error=policy["maximum_fixed_level_gpu_error"],
        minimum_worst_error_improvement_factor=policy[
            "minimum_worst_error_improvement_factor"
        ],
        selection_target_diagnostic_only=0.05,
        gpu_executed=False,
    ),
)
assert gpu["passed"] and gpu["complete"] and len(gpu["rows"]) == 212
assert (
    reference["passed"]
    and reference["source_unchanged"]
    and reference["reference_refinement_passed"]
)
assert sha256(baseline_path) == policy["fixed_node_baseline"]["sha256"]
baseline = read_json(baseline_path)
assert baseline["passed"] and all(
    x["active_contacts"] == x["reference_active_contacts"] for x in baseline["rows"]
)
assert len(baseline["rows"]) == 106
source = gpu_path.parent / "source_snapshot/src/sand_atlas_dem/spec.py"
assert sha256(source) == gpu["source_hashes"]["src/sand_atlas_dem/spec.py"]
model = next(
    x
    for x in ast.parse(source.read_text()).body
    if isinstance(x, ast.ClassDef) and x.name == "Experiment"
)
kn = ast.literal_eval(
    next(
        x.value
        for x in model.body
        if isinstance(x, ast.AnnAssign) and x.target.id == "normal_stiffness"
    )
)
refs = {(r["shape"], r["separation_m"]): r for r in reference["rows"]}
assert len(refs) == 106
rows = []
for old in gpu["rows"]:
    assert old["status"] == "complete"
    r = refs[(old["shape"], old["separation_m"])]
    level = old["subdivisions"]
    assert level < r["reference_finest_level"]
    observed = np.array(old["force_N"])
    final = kn * np.array(r["levels"][-1]["force_integral_m3"])
    fixed = kn * np.array(r["levels"][level]["force_integral_m3"])
    fnorm = np.linalg.norm(fixed)
    if fnorm == 0:
        assert (
            np.array_equal(observed, np.zeros(3))
            and old["active_contacts"] == r["levels"][level]["active_nodes"] == 0
        )
        fixed_error = 0.0
    else:
        fixed_error = float(np.linalg.norm(observed - fixed) / fnorm)
    error = float(np.linalg.norm(observed - final) / np.linalg.norm(final))
    rows.append(
        dict(
            library_index=old["library_index"],
            shape=old["shape"],
            separation_m=old["separation_m"],
            subdivisions=level,
            reference_finest_level=r["reference_finest_level"],
            relative_force_error=error,
            fixed_level_gpu_relative_error=fixed_error,
        )
    )
coarse = [r for r in rows if r["library_index"] == 0]
fine = [r for r in rows if r["library_index"] == 1]
assert len(coarse) == len(fine) == 106
coarse_worst = max(r["relative_force_error"] for r in coarse)
fine_worst = max(r["relative_force_error"] for r in fine)
fixed_worst = max(r["fixed_level_gpu_relative_error"] for r in rows)
passed = bool(
    fine_worst <= policy["maximum_relative_force_error"]
    and fixed_worst <= policy["maximum_fixed_level_gpu_error"]
    and fine_worst * policy["minimum_worst_error_improvement_factor"] <= coarse_worst
)
assert all(sha256(p) == digest for p, digest in inputs.items())
write_json(
    out / "report.json",
    dict(
        identity=out.name,
        passed=passed,
        source_unchanged=True,
        gpu_executed=False,
        reanalyzed_existing_gpu_evidence=True,
        scientific_admission=False,
        completed_cases=len(rows),
        normal_traction_stiffness_N_per_m3=kn,
        coarse_worst_relative_error=coarse_worst,
        fine_worst_relative_error=fine_worst,
        maximum_fixed_level_gpu_error=fixed_worst,
        fine_cases_above_preprocessing_selection_target=[
            r for r in fine if r["relative_force_error"] > 0.05
        ],
        input_sha256=inputs,
        rows=rows,
        scope="Retrospective application of unchanged original10% integral/0.1% same-level/twofold gates to all212 retained force observations and stronger same-pose references. This does not establish5% preprocessing support for the old asset, dynamic accuracy or material admission. Original GPU study remains untouched.",
    ),
)
