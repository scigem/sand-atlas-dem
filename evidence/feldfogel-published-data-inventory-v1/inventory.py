"""Read-only inventory of the pinned Feldfogel benchmark source archive."""

import argparse
import hashlib
import io
import shutil
import zipfile
from pathlib import Path

import numpy as np

from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = read_json(args.protocol)
    protocol_hash = sha256(args.protocol)
    for item in policy["sources"]:
        raw = Path(item["path"]).read_bytes()
        if (
            hashlib.sha256(raw).hexdigest() != item["sha256"]
            or hashlib.md5(raw).hexdigest() != item["publisher_md5"]
        ):
            raise ValueError("Source checksum changed")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, args.output / "protocol.json")
    shutil.copy2(__file__, args.output / "inventory.py")
    code_hash = sha256(__file__)
    write_json(
        args.output / "prospective-freeze.json",
        dict(protocol_sha256=protocol_hash, inventory_source_sha256=code_hash),
    )
    members = []
    with zipfile.ZipFile(policy["archive"]) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            raw = archive.read(entry)
            text = raw.decode("utf-8")
            # The literal source header is recorded, never inferred as data.
            header = text.splitlines()[0]
            skip = int(header == "dum")
            values = np.loadtxt(io.StringIO(text), skiprows=skip, ndmin=2)
            members.append(
                dict(
                    name=entry.filename,
                    size_bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    source_header=header if skip else None,
                    rows=values.shape[0],
                    columns=values.shape[1],
                    finite=bool(np.isfinite(values).all()),
                    minimum_by_column=values.min(axis=0),
                    maximum_by_column=values.max(axis=0),
                )
            )
    duplicate_groups = {}
    for member in members:
        duplicate_groups.setdefault(member["sha256"], []).append(member["name"])
    if sha256(__file__) != code_hash or sha256(args.protocol) != protocol_hash:
        raise ValueError("Inventory source or protocol changed")
    for item in policy["sources"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("Source changed during inventory")
    write_json(
        args.output / "report.json",
        dict(
            protocol=policy,
            protocol_sha256=protocol_hash,
            inventory_source_sha256=code_hash,
            members=members,
            identical_member_groups=[
                v for v in duplicate_groups.values() if len(v) > 1
            ],
            inventory_complete=True,
            all_numeric_values_finite=all(m["finite"] for m in members),
            scientific_admission=False,
            scope="Source-byte and numerical-array inventory only. No simulation, curve-column assignment, parameter calibration, benchmark reproduction or validation.",
        ),
    )


if __name__ == "__main__":
    main()
