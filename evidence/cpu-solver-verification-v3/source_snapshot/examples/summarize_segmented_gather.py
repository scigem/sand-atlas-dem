"""Audit frozen contiguous-gather archives and retain their failed gate decisions."""

import json
from pathlib import Path

from sand_atlas_dem.util import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "evidence/segmented-gather-summary-v1"
    out.mkdir(parents=True, exist_ok=False)
    studies = [
        "dense27-segmented-gather-v1",
        "hostun-segmented-gather-v1",
        "dense27-zero-segmented-gather-v1",
    ]
    probes = ["mesh-face-normal-probe-v1", "mesh-face-normal-probe-v2"]
    paths = [
        ROOT / "evidence" / name / file
        for name in studies
        for file in ["report.json", "protocol.json", "prospective-freeze.json"]
    ]
    paths += [
        ROOT / "evidence" / name / file
        for name in probes
        for file in ["report.json", "protocol.json", "driver.py"]
    ]
    paths += [Path(__file__)]
    write_json(
        out / "analysis-inputs.json",
        {str(p.relative_to(ROOT)): sha256(p) for p in paths},
    )
    (out / "driver.py").write_bytes(Path(__file__).read_bytes())
    audits = {}
    results = {}
    for name in studies:
        directory = ROOT / "evidence" / name
        frozen = json.loads((directory / "prospective-freeze.json").read_text())
        checks = {
            str(p): sha256(directory / "source_snapshot" / p) == digest
            for p, digest in frozen["sources"].items()
        }
        checks.update(
            {str(p): sha256(p) == digest for p, digest in frozen["inputs"].items()}
        )
        checks["protocol"] = (
            sha256(directory / "protocol.json") == frozen["protocol_sha256"]
        )
        audits[name] = dict(passed=all(checks.values()), checks=checks)
        report = json.loads((directory / "report.json").read_text())
        checks = report["checks"]
        results[name] = dict(
            medians=report["median_seconds_per_step"],
            speedups=report["speedups"],
            failed_checks=[c for c in checks if not c["passed"]],
            equivalence_passed=report["equivalence_passed"],
            production_promoted=False,
        )
    for name in probes:
        directory = ROOT / "evidence" / name
        p = json.loads((directory / "protocol.json").read_text())
        checks = {
            module: sha256(module) == digest for module, digest in p["modules"].items()
        }
        checks["driver"] = sha256(directory / "driver.py") == p["driver_sha256"]
        audits[name] = dict(
            passed=all(checks.values()),
            checks=checks,
            invalid_driver=name.endswith("v1"),
        )
    write_json(out / "archive-check.json", audits)
    write_json(
        out / "summary.json",
        dict(studies=results, scientific_admission=False, production_promoted=False),
    )
    assert all(a["passed"] for a in audits.values())
    print({name: value["passed"] for name, value in audits.items()})


if __name__ == "__main__":
    main()
