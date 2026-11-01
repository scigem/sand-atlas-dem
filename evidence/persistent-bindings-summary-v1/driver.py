"""Audit canonical-order studies without replacing or waiving failed comparisons."""

import json
from pathlib import Path

from sand_atlas_dem.util import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
STUDIES = [
    "dense27-spin-persistent-slots-v1",
    "dense27-spin-fused-slots-v1",
    "dense27-gather-submission-profile-v1",
    "dense27-immutable-bindings-v1",
    "hostun-immutable-bindings-v1",
]
PROBES = [
    "persistent-slot-transition-probe-v1",
    "fused-slot-reduction-probe-v1",
    "immutable-binding-transition-probe-v1",
]


def main():
    out = ROOT / "evidence/persistent-bindings-summary-v1"
    out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__)]
    for name in STUDIES + PROBES:
        directory = ROOT / "evidence" / name
        paths.extend(
            p for p in directory.iterdir() if p.is_file() and p.suffix == ".json"
        )
    write_json(
        out / "analysis-inputs.json",
        {str(p.relative_to(ROOT)): sha256(p) for p in paths},
    )
    (out / "driver.py").write_bytes(Path(__file__).read_bytes())
    audits = {}
    results = {}
    for name in STUDIES:
        directory = ROOT / "evidence" / name
        f = json.loads((directory / "prospective-freeze.json").read_text())
        checks = {
            p: sha256(directory / "source_snapshot" / p) == h
            for p, h in f["sources"].items()
        }
        checks.update({p: sha256(p) == h for p, h in f["inputs"].items()})
        checks["protocol"] = sha256(directory / "protocol.json") == f["protocol_sha256"]
        audits[name] = dict(passed=all(checks.values()), checks=checks)
        if not (directory / "report.json").exists():
            results[name] = dict(
                completed=False,
                failure=json.loads((directory / "failure-receipt.json").read_text()),
            )
            continue
        r = json.loads((directory / "report.json").read_text())
        repeat = {}
        for mode in r["median_seconds_per_step"]:
            checks = [
                c for c in r["checks"] if c["mode"] == mode and "reference" not in c
            ]
            repeat[mode] = dict(
                passed=sum(c["passed"] for c in checks),
                total=len(checks),
                all_bitwise_body_fields=all(
                    all(c["bitwise_body_state"].values()) for c in checks
                ),
            )
        results[name] = dict(
            completed=True,
            medians=r["median_seconds_per_step"],
            speedups=r["speedups"],
            equivalence_passed=r["equivalence_passed"],
            repeat=repeat,
            failed_checks=[c for c in r["checks"] if not c["passed"]],
            phase_medians=r.get("phase_medians"),
            binding_cache_counts=r.get("binding_cache_counts"),
            target_2x=r.get("target_2x"),
            production_promoted=False,
        )
    for name in PROBES:
        directory = ROOT / "evidence" / name
        p = json.loads((directory / "protocol.json").read_text())
        if "module" in p:
            checks = dict(
                module=sha256(p["module"]) == p["module_sha256"],
                driver=sha256(directory / "driver.py") == p["driver_sha256"],
            )
        else:
            checks = dict(
                module=sha256(directory / "probe.slang") == p["shader_sha256"],
                driver=sha256(directory / "driver.py") == p["driver_sha256"],
                helper=sha256(directory / "immutable_bindings.py")
                == p["helper_sha256"],
            )
        audits[name] = dict(passed=all(checks.values()), checks=checks)
        results[name] = (
            json.loads((directory / "report.json").read_text())
            if (directory / "report.json").exists()
            else dict(
                completed=False,
                failure=json.loads((directory / "failure-receipt.json").read_text()),
            )
        )
    write_json(out / "archive-check.json", audits)
    write_json(
        out / "summary.json",
        dict(results=results, production_promoted=False, scientific_admission=False),
    )
    assert all(a["passed"] for a in audits.values())
    print(dict(audited=len(audits), all_unchanged=True))


if __name__ == "__main__":
    main()
