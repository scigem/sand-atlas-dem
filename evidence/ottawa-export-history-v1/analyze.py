"""Read immutable local Git blobs; do not execute producer source or admit units."""

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


OUT = Path(__file__).resolve().parent
REPO = Path("/Users/bmar5496/code/sand-atlas-python")
ARCHIVE = OUT.parent / "ottawa1um-source-intake-v1/private/meshes_30.zip"
SOURCE = "sand_atlas/blender_scripts/vdb.py"
REVISIONS = [
    (
        "a8be484b90a5d29f43e2ffc656b94d3c5ea7ea31",
        "voxel_size_m times volume correction",
    ),
    (
        "141a7785f84dc4eb58d82a3a0648e70a0b3e7c7f",
        "volume correction alone; voxel coordinates",
    ),
    (
        "e902cc5d2e373cbce3e84ba977bf8eafa2b598fc",
        "voxel_size_m times 1000 times volume correction",
    ),
]


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    own_hash = digest(Path(__file__).read_bytes())
    archive_hash = digest(ARCHIVE.read_bytes())
    blobs = [
        (revision, interpretation, git("show", f"{revision}:{SOURCE}"))
        for revision, interpretation in REVISIONS
    ]
    freeze = dict(
        script_sha256=own_hash,
        archive_sha256=archive_hash,
        blobs={revision: digest(data) for revision, _, data in blobs},
        scope=__doc__,
        admission="No gate or historical-producer assignment. ZIP dates are nonbinding metadata.",
    )
    (OUT / "prospective-freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    rows = []
    for revision, interpretation, data in blobs:
        lines = data.decode().splitlines()
        rows.append(
            dict(
                commit=revision,
                blob=git("rev-parse", f"{revision}:{SOURCE}").decode().strip(),
                source_sha256=digest(data),
                source_path=SOURCE,
                commit_date=git("show", "-s", "--format=%aI", revision)
                .decode()
                .strip(),
                url=f"https://github.com/SciGEM/sand-atlas-python/blob/{revision}/{SOURCE}",
                active_scale_assignment_lines=[
                    i
                    for i, line in enumerate(lines, 1)
                    if line.strip().startswith(("obj.scale =", "voxel_size_mm ="))
                ],
                observed_scale_expression=interpretation,
                executed=False,
            )
        )
    with zipfile.ZipFile(ARCHIVE) as archive:
        dates = sorted({tuple(i.date_time[:3]) for i in archive.infolist()})
        members = len(archive.infolist())
    assert digest(Path(__file__).read_bytes()) == own_hash
    assert digest(ARCHIVE.read_bytes()) == archive_hash
    assert all(
        digest(git("show", f"{r}:{SOURCE}")) == freeze["blobs"][r] for r, _, _ in blobs
    )
    report = dict(
        identity=OUT.name,
        inputs_unchanged=True,
        producer_revisions=rows,
        zip_member_dates=dates,
        archive_members=members,
        original_export_receipt_found=False,
        coordinate_units_admitted=False,
        material_registered=False,
        gpu_executed=False,
        scientific_admission=False,
        conclusion="Historical exporter units changed. ZIP metadata dates do not bind exporter commits, label maps, argument values, Blender unit settings or regeneration history. No historical producer can be selected from this evidence alone.",
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
