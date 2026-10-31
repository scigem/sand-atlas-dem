"""Freeze and check the public Atlas bridge on an OWN ANALYTIC FORMAT FIXTURE.

No DEM specimen or constitutive engine is executed. This verifies mapping and the
actual Python/browser request consumers; it does not qualify a material response.
Requires the development tests (pytest) and Node, alongside both local checkouts.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "private").mkdir(mode=0o700)
    files = [
        "src/sand_atlas_dem/atlas_bridge.py",
        "src/sand_atlas_dem/cli.py",
        "src/sand_atlas_dem/cases.py",
        "tests/test_atlas_bridge.py",
        "cases/atlas-projection.json",
        "cases/atlas-elastic-replay.json",
        "examples/atlas_bridge_study.py",
        "src/sand_atlas_dem/geometry.py",
        "src/sand_atlas_dem/spec.py",
        "src/sand_atlas_dem/util.py",
    ]
    frozen = {name: sha256(root / name) for name in files}
    for name in files:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    write_json(output / "prospective-freeze.json", dict(scope=__doc__, files=frozen))
    module_spec = importlib.util.spec_from_file_location(
        "bridge_fixture", root / "tests/test_atlas_bridge.py"
    )
    fixture = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(fixture)
    run, policy = fixture.run_fixture.__wrapped__(output / "private")
    command = [
        sys.executable,
        "-m",
        "sand_atlas_dem.cli",
        "atlas",
        "prepare",
        str(run),
        "--policy",
        str(policy),
        "--configuration",
        str(root / "cases/atlas-elastic-replay.json"),
        "--atlas-root",
        str(args.atlas_root.resolve()),
        "--output",
        str(output / "handoff"),
    ]
    with (output / "cli.log").open("w") as log:
        subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env={**os.environ, "PYTHONPATH": str(root / "src")},
        )
    request = read_json(output / "handoff/replay-request.json")
    virtual = read_json(output / "handoff/virtual-test.json")
    assert virtual["kind"] == "synthetic-dem" and not virtual["scientific_admission"]
    assert len(request["path"]["knots"]) == 5
    targets = [r["targets"][0] for r in request["path"]["knots"]]
    assert targets[0] == 0 and targets[2] > targets[3] and targets[3] == targets[4]
    node = """import {readFileSync} from 'node:fs';import {pathToFileURL} from 'node:url';
    const {parseRecordedJSON,validateRecordedRun}=await import(pathToFileURL(process.argv[1]));
    validateRecordedRun(parseRecordedJSON(readFileSync(process.argv[2],'utf8')));"""
    with (output / "browser-contract.log").open("w") as log:
        subprocess.run(
            [
                "node",
                "--input-type=module",
                "-e",
                node,
                str(args.atlas_root.resolve() / "web/recorded.mjs"),
                str(output / "handoff/replay-request.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    assert all(sha256(root / name) == digest for name, digest in frozen.items())
    write_json(
        output / "report.json",
        dict(
            format="sand-dem-atlas-bridge-software-check/1",
            passed=True,
            scope=__doc__,
            scientific_admission=False,
            files=frozen,
            handoff_sha256=sha256(output / "handoff/handoff.json"),
            node_version=subprocess.check_output(
                ["node", "--version"], text=True
            ).strip(),
            observation_count=5,
            checks=[
                "Actual public CLI",
                "Actual Atlas Python prepare-replay",
                "Actual browser recorded-request parser/validator",
                "Prepared origin and preserved reversal/duplicate",
                "Explicit synthetic evidence identity",
                "Executed source freeze unchanged",
            ],
        ),
    )


if __name__ == "__main__":
    main()
