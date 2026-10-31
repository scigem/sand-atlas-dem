"""Freeze sources and run the serial actual-Metal parallel mesh qualification suite."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sand_atlas_dem.util import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = sorted(
        p
        for directory in ["src", "tests", "examples", "studies", "cases"]
        for p in (root / directory).rglob("*")
        if p.is_file() and p.suffix in {".py", ".slang", ".json"}
    )
    paths.append(root / "pyproject.toml")
    hashes = {}
    for p in paths:
        key = str(p.relative_to(root))
        hashes[key] = sha256(p)
        dest = out / "source_snapshot" / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    write_json(
        out / "prospective-freeze.json",
        dict(
            format="parallel-mesh-qualification/1",
            files=hashes,
            policy="32 disjoint complete source subtrees per directed pair, GPU root mask and indirect dispatch",
            gates="tests/test_parallel_mesh.py, frozen source before execution",
            backend="Metal",
            full_suite=True,
        ),
    )
    env = dict(os.environ, SAND_DEM_METAL="1", PYTHONPATH=str(root / "src"))
    with (out / "pytest.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--junitxml=" + str(out / "pytest.xml"),
                "-o",
                "junit_family=xunit1",
            ],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    unchanged = all(
        sha256(root / p) == h and sha256(out / "source_snapshot" / p) == h
        for p, h in hashes.items()
    )
    write_json(
        out / "report.json",
        dict(
            returncode=result.returncode,
            source_integrity=unchanged,
            files=len(hashes),
            software_execution_passed=result.returncode == 0 and unchanged,
            scientific_admission=False,
        ),
    )
    if result.returncode or not unchanged:
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
