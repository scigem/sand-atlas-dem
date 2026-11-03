"""Freeze a release and run regression, independent contact and mechanics studies."""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sand_atlas_dem.util import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    env = dict(os.environ, SAND_DEM_METAL="1", PYTHONPATH=str(root / "src"))
    required = (
        "SAND_DEM_ATLAS_ROOT",
        "SAND_DEM_ATLAS_SOLVER",
        "SAND_DEM_ATLAS_CATALOGUE",
    )
    if any(not env.get(key) or not Path(env[key]).exists() for key in required):
        parser.error(
            "Set existing Atlas checkout, solver and catalogue paths explicitly"
        )
    out.mkdir(parents=True, exist_ok=False)
    paths = sorted(
        p
        for directory in ("src", "tests", "examples", "studies", "cases")
        for p in (root / directory).rglob("*")
        if p.is_file() and p.suffix in {".py", ".slang", ".json"}
    ) + [root / "pyproject.toml"]
    hashes = {str(p.relative_to(root)): sha256(p) for p in paths}
    for p in paths:
        dest = out / "source_snapshot" / p.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    # Hash private fixtures without copying geometry into distributable evidence.
    external = [Path(env[key]).resolve() for key in required[1:]]
    external += sorted((root / "assets/generated").glob("*.h5"))
    inputs = {str(p): sha256(p) for p in external}
    commands = {
        "pytest": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-ra",
            "--junitxml=" + str(out / "pytest.xml"),
        ],
        "contact-refinement": [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, 'examples'); from contact_refinement import study; "
                "raise SystemExit(study(sys.argv[1]))"
            ),
            str(out / "contact-refinement.json"),
        ],
        "mechanics": [
            sys.executable,
            "-c",
            "from sand_atlas_dem.cli import main; raise SystemExit(main())",
            "qualify",
            "studies/mechanics-v2.json",
            "--output",
            str(out / "mechanics"),
        ],
    }
    freeze = dict(
        format="sand-dem-release-qualification/1",
        backend="Metal",
        files=hashes,
        inputs=inputs,
        commands=commands,
        python=sys.version,
        platform=platform.platform(),
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        atlas_environment={key: env[key] for key in required},
        policy="Serial full suite with Atlas; existing contact-refinement 2% and mechanics-v2 gates unchanged. No scientific admission.",
    )
    write_json(out / "prospective-freeze.json", freeze)
    results = {}
    for name, command in commands.items():
        print("Starting", name, flush=True)
        start = time.monotonic()
        with (out / (name + ".log")).open("w") as log:
            result = subprocess.run(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        results[name] = dict(
            returncode=result.returncode, seconds=time.monotonic() - start
        )
        print(name, results[name], flush=True)
        write_json(out / "progress-report.json", results)
    unchanged = all(
        sha256(root / p) == h and sha256(out / "source_snapshot" / p) == h
        for p, h in hashes.items()
    )
    inputs_unchanged = all(sha256(p) == h for p, h in inputs.items())
    passed = (
        unchanged
        and inputs_unchanged
        and all(r["returncode"] == 0 for r in results.values())
    )
    write_json(
        out / "report.json",
        dict(
            commit=freeze["commit"],
            results=results,
            source_integrity=unchanged,
            input_integrity=inputs_unchanged,
            software_execution_passed=passed,
            scientific_admission=False,
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
