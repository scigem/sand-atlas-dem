"""Independent CPU inertia fingerprint diagnostic; no identity/scale admission."""

import csv
import io
from pathlib import Path
import zipfile

import numpy as np
import trimesh

from sand_atlas_dem.util import read_json, sha256, write_json


def full_axes(volume, inertia):
    """Volume-enforced inertia ellipsoid, derived from central second moments."""
    moments = np.linalg.eigvalsh(inertia)
    covariance = (0.5 * np.sum(moments) - moments) / volume
    if not np.all(covariance > 0):
        raise ValueError("Nonpositive central second moment")
    axes = 2 * np.sqrt(5 * covariance)
    return np.sort(axes * (6 * volume / (np.pi * np.prod(axes))) ** (1 / 3))[::-1]


def main():
    out = Path(__file__).resolve().parent
    source = out.parent / "ottawa1um-source-intake-v1"
    archive = source / "private/meshes_30.zip"
    table = source / "source_snapshot/sand-atlas/assets/sands/ottawa-0000008.csv"
    previous = out.parent / "ottawa1um-source-intake-v2/report.json"
    inputs = [archive, table, previous, out / "source-receipt.json",
              out / "private/spam-label-source.html", Path(__file__)]
    hashes = {str(p): sha256(p) for p in inputs}
    write_json(out / "prospective-freeze.json", dict(
        input_sha256=hashes, numpy=np.__version__, trimesh=trimesh.__version__,
        scope=__doc__, matching="Rank all descriptors by maximum absolute log ratio across raw volume and three full axes; retain top two, no fitted thresholds or global reassignment.",
        limitations="Current SPAM convention is not proof of the historical producer version; axes and volume are not statistically independent after volume enforcement."))
    # Independent box formula, including translated/rotated geometry.
    sides = np.array([2., 3., 5.])
    expected = (6 / np.pi) ** (1 / 3) * np.sort(sides)[::-1]
    box = trimesh.creation.box(extents=sides)
    first_error = float(np.max(np.abs(full_axes(box.volume, box.moment_inertia) / expected - 1)))
    transform = trimesh.transformations.rotation_matrix(0.73, [1, 2, 3])
    transform[:3, 3] = [17., -4., 2.]
    box.apply_transform(transform)
    moved_error = float(np.max(np.abs(full_axes(box.volume, box.moment_inertia) / expected - 1)))
    assert max(first_error, moved_error) < 1e-12
    with table.open(newline="") as f:
        descriptors = list(csv.DictReader(f))
    columns = ["Volume (µm³)", "Major Axis Length (µm)",
               "Middle Axis Length (µm)", "Minor Axis Length (µm)"]
    features = np.array([[float(d[c]) for c in columns] for d in descriptors])
    assert np.all(np.isfinite(features)) and np.all(features > 0)
    algebra = np.pi / 6 * np.prod(features[:, 1:], axis=1) / features[:, 0] - 1
    old = {r["member"]: r for r in read_json(previous)["rows"]}
    rows = []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            prior = old[info.filename]
            row = dict(member=info.filename, basic_topology_supported=prior["basic_topology_supported"])
            if prior["basic_topology_supported"]:
                mesh = trimesh.load_mesh(io.BytesIO(z.read(info)), file_type="stl", process=False)
                axes = full_axes(mesh.volume, mesh.moment_inertia)
                observed = np.r_[mesh.volume, axes]
                errors = np.log(observed / features)
                scores = np.max(np.abs(errors), axis=1)
                order = np.argsort(scores, kind="stable")
                row.update(raw_full_axes=axes.tolist(), top_two=[dict(
                    descriptor_id=descriptors[int(i)]["Particle ID"],
                    maximum_absolute_log_ratio=float(scores[i]),
                    relative_feature_differences=(observed / features[i] - 1).tolist()) for i in order[:2]],
                    prior_volume_nearest_id=prior["closest_descriptor_id"])
            rows.append(row)
    ids = [r["top_two"][0]["descriptor_id"] for r in rows if "top_two" in r]
    assert all(sha256(p) == h for p, h in hashes.items())
    report = dict(identity=out.name, input_sha256=hashes, inputs_unchanged=True,
                  analytic_box_max_relative_error=first_error,
                  transformed_box_max_relative_error=moved_error,
                  descriptor_ellipsoid_volume_max_relative_error=float(np.max(np.abs(algebra))),
                  matched_meshes=len(ids), unique_nearest_assignment=len(set(ids)) == len(ids),
                  duplicate_nearest_ids=sorted({i for i in ids if ids.count(i) > 1}, key=int),
                  maximum_best_score=max(r["top_two"][0]["maximum_absolute_log_ratio"] for r in rows if "top_two" in r),
                  rows=rows, coordinate_units_admitted=False, material_registered=False,
                  gpu_executed=False, scientific_admission=False)
    write_json(out / "report.json", report)
    print({k: v for k, v in report.items() if k not in ("rows", "input_sha256")})
    print([r for r in rows if r["member"] in ("particle_00054.stl", "particle_00084.stl")])


if __name__ == "__main__":
    main()
