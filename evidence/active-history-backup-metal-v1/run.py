"""Run the frozen device checks only after the existing preparation is terminal."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    policy = json.loads((root / "prospective-freeze.json").read_text())
    for name, expected in policy["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or digest(path) != expected:
            raise ValueError(f"Frozen input mismatch: {name}")
    if not args.execute:
        print("Frozen inputs verified; no GPU initialized or test executed.")
        return 0
    # This is a dependency check, not proof that no unrelated GPU job exists.
    dependency = json.loads(Path(policy["preparation_manifest"]).read_text())
    if dependency["status"] not in ("complete", "failed"):
        raise RuntimeError(
            "Existing preparation is not terminal; keep GPU tests queued"
        )
    output = root / "execution"
    output.mkdir()  # Never overwrite a previous attempt, including a failed one.
    env = dict(
        os.environ, PYTHONPATH=str(root / "source_snapshot/src"), SAND_DEM_METAL="1"
    )
    command = [sys.executable, "-m", "pytest", "-q", *policy["test_nodes"]]
    with (output / "pytest.log").open("w") as log:
        result = subprocess.run(
            command,
            cwd=root / "source_snapshot",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    unchanged = all(
        digest(root / name) == expected for name, expected in policy["files"].items()
    )
    (output / "report.json").write_text(
        json.dumps(
            dict(
                exit_code=result.returncode,
                sources_unchanged=unchanged,
                log_sha256=digest(output / "pytest.log"),
                scope="Actual-device test attempt; consult pytest log for individual pass/skip/failure counts. No dense benchmark or scientific admission.",
                scientific_admission=False,
            ),
            indent=2,
        )
        + "\n"
    )
    return result.returncode if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
