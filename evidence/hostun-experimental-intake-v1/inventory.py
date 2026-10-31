"""Read-only evidence intake; no experimental state/strain normalization or fitting."""

import gzip
from pathlib import Path

import numpy as np

from sand_atlas_dem.util import read_json, sha256, write_json

out = Path(__file__).resolve().parent
atlas = Path("/Users/bmar5496/code/constitutive-atlas")
sand = Path("/Users/bmar5496/code/sand-atlas")
paths = sorted((atlas / "data/curated").glob("GEOLAB-HOSTUN-TRIAX-*.json.gz"))
packets = [
    atlas / "paper/study" / name
    for name in ("hostun_feasibility.json", "hostun_source_metadata.json")
]
metadata_path = sand / "_data/json/hostun-0000005.json"
inputs = {str(p): sha256(p) for p in [*paths, *packets, metadata_path, Path(__file__)]}
rows = []
for path in paths:
    import json

    data = json.loads(gzip.decompress(path.read_bytes()))
    columns = data["columns"]
    axial = next(c["key"] for c in columns if c["name"] == "Axial strain")
    stress = next(c["key"] for c in columns if c["name"] == "Deviatoric stress")
    points = [(r[axial], r[stress]) for r in data["rows"] if axial in r and stress in r]
    delta = np.diff(np.array(points)[:, 0])
    rows.append(
        dict(
            id=data["id"],
            path=str(path),
            sha256=sha256(path),
            columns=columns,
            initial=data["initial"],
            conventions=data["conventions"],
            native_complete_pairs=len(points),
            consecutive_duplicate_native_abscissae=int(np.count_nonzero(delta == 0)),
            native_abscissa_decreases=int(np.count_nonzero(delta < 0)),
            source_url=data["provenance"]["source_url"],
            workbook_sha256=data["provenance"]["archive_sha256"],
            selected_cells_sha256=data["provenance"]["member_sha256"],
            warnings=data["warnings"],
            normalized=False,
            admitted_for_calibration=False,
        )
    )
if any(sha256(path) != digest for path, digest in inputs.items()):
    raise ValueError("External evidence changed during read-only inventory")
metadata = read_json(metadata_path)
write_json(
    out / "report.json",
    dict(
        identity=out.name,
        source_unchanged=True,
        input_sha256=inputs,
        series_count=len(rows),
        series=rows,
        shape_material_source_url=metadata["publication_url"],
        geometry_to_experimental_batch_identity_established=False,
        scientific_admission=False,
        calibration_executed=False,
        normalization_executed=False,
        retained_prior_atlas_review_ids=[read_json(p)["id"] for p in packets],
        scope="Native read-only intake of existing Atlas Hostun records and prior source-conflict packets. All points remain untouched. No source channel relabeling, state inference, curve joining, calibration/validation split or fit. A shared material name does not identify the same physical batch.",
    ),
)
