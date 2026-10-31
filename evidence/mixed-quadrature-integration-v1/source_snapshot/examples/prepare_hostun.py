"""Private source ingestion; keeps originals, rejects invalid solids, never repairs."""

import concurrent.futures as cf
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.util import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "assets/private/hostun-meshes-30.zip"
CSV = ROOT / "assets/private/hostun-descriptors.csv"


def one(task):
    name, blob, descriptor, res, samples = task
    m = trimesh.load_mesh(io.BytesIO(blob), file_type="stl", process=False)
    v, ix = np.unique(m.vertices, axis=0, return_inverse=True)
    m = trimesh.Trimesh(v, ix[m.faces], process=False)
    parts = m.split(only_watertight=False)
    if (
        len(parts) != 1
        or not m.is_watertight
        or not m.is_winding_consistent
        or m.volume <= 0
    ):
        return (
            name,
            None,
            dict(
                status="rejected",
                components=len(parts),
                reason="not_single_closed_consistent_positive_solid",
            ),
        )
    target = (
        ROOT
        / "assets/private"
        / f"hostun-30-r{res}-s{samples}"
        / f"{Path(name).stem}.h5"
    )
    provenance = dict(
        dataset="Hostun sand",
        source_url="https://data.scigem-eng.sydney.edu.au/sand-atlas/sands/hostun-0000005/meshes_30.zip",
        source_archive_sha256=sha256(ARCHIVE),
        archive_member=name,
        source_sha256=hashlib.sha256(blob).hexdigest(),
        descriptor_table_sha256=sha256(CSV),
        descriptor_id=descriptor["Particle ID"],
        unit_basis="Inferred from unique one-to-one raw-volume agreement with source table Volume (µm³); table/mesh IDs differ. One raw coordinate unit is 1 micrometre; source metadata voxel size 1 µm. See hostun-acquisition-v2 report.",
        repairs=[],
    )
    start = time.perf_counter()
    if target.exists():
        shape = ShapeLibrary.load(target).shapes[0]
        if shape.provenance.get("source_sha256") != provenance["source_sha256"]:
            raise ValueError("Cached source changed")
    else:
        shape = from_mesh(
            m,
            shape_id="hostun-" + Path(name).stem,
            unit="um",
            resolution=res,
            samples=samples,
            provenance=provenance,
        )
        ShapeLibrary([shape]).save(target)
    return (
        name,
        str(target),
        dict(
            status="prepared",
            volume_m3=shape.volume,
            radius_m=shape.radius,
            grid=list(shape.sdf.shape),
            spacing_m=shape.spacing,
            wall_seconds=time.perf_counter() - start,
            descriptor_id=descriptor["Particle ID"],
            source_sha256=provenance["source_sha256"],
            shape_sha256=sha256(target),
        ),
    )


def main():
    res = int(sys.argv[1])
    samples = int(sys.argv[2])
    workers = int(sys.argv[3])
    out = ROOT / "assets/private" / f"hostun-30-r{res}-s{samples}"
    out.mkdir(parents=True, exist_ok=True)
    descriptors = list(csv.DictReader(CSV.open()))
    topology = json.loads(
        (ROOT / "evidence/hostun-archive-topology-v2.json").read_text()
    )
    tasks = []
    mappings = []
    with zipfile.ZipFile(ARCHIVE) as z:
        for r in topology:
            choices = sorted(
                [
                    (abs(r["volume"] / float(d["Volume (µm³)"]) - 1), d)
                    for d in descriptors
                ],
                key=lambda x: x[0],
            )
            error, d = choices[0]
            mappings.append(
                dict(
                    member=r["name"],
                    descriptor_id=d["Particle ID"],
                    relative_volume_difference=error,
                    next_best_difference=choices[1][0],
                )
            )
            tasks.append((r["name"], z.read(r["name"]), d, res, samples))
    if len(set(r["descriptor_id"] for r in mappings)) != len(mappings):
        raise ValueError("Nonunique descriptor mapping")
    # This is source unit inference with explicit residuals, not an experimental validation gate.
    report = dict(
        format="sand-dem-hostun-ingestion/1",
        archive_sha256=sha256(ARCHIVE),
        descriptor_sha256=sha256(CSV),
        coordinate_unit="um",
        unit_status="inferred_from_dimensional_source_descriptors",
        mappings=mappings,
        resolution=res,
        samples=samples,
        repairs=[],
        grains={},
        status="running",
    )
    evidence = ROOT / "evidence" / f"hostun-ingestion-r{res}-s{samples}-v1.json"
    write_json(evidence, report)
    with cf.ProcessPoolExecutor(max_workers=workers) as executor:
        for name, path, record in executor.map(one, tasks):
            report["grains"][name] = dict(record, path=path)
            write_json(evidence, report)
            print(
                name,
                record["status"],
                round(record.get("wall_seconds", 0), 2),
                flush=True,
            )
    paths = [r["path"] for r in report["grains"].values() if r["status"] == "prepared"]
    combined = ROOT / "assets/private" / f"hostun-30-r{res}-s{samples}.h5"
    ShapeLibrary([ShapeLibrary.load(p).shapes[0] for p in paths]).save(combined)
    report.update(
        status="complete",
        accepted_grains=len(paths),
        library_path=str(combined),
        library_sha256=sha256(combined),
    )
    write_json(evidence, report)


if __name__ == "__main__":
    main()
