"""Case resolution is CPU-only; execution failures retain their identity."""

import os
import sys

import pytest

from sand_atlas_dem.cases import (
    register_material,
    resolve_case,
    resolve_material,
    run_case,
)
from sand_atlas_dem.cli import main
from sand_atlas_dem.geometry import ShapeLibrary, sphere
from sand_atlas_dem.util import read_json, sha256, write_json


@pytest.fixture
def inputs(tmp_path):
    library = tmp_path / "shape.h5"
    ShapeLibrary([sphere(resolution=16, samples=64)]).save(library)
    receipt = tmp_path / "receipt.json"
    write_json(
        receipt,
        dict(
            format="sand-dem-material/1",
            id="analytical-fixture",
            aliases=["fixture"],
            shape_library="shape.h5",
            shape_sha256=sha256(library),
            source={"kind": "analytical-sphere-fixture"},
            scale_evidence="Analytical radius explicitly specified in metres; not Sand Atlas data.",
        ),
    )
    cache = tmp_path / "cache"
    register_material(receipt, cache)
    case = tmp_path / "case.json"
    write_json(
        case,
        dict(
            format="sand-dem-case/1",
            experiment=dict(
                particles=27,
                density=2650,
                normal_stiffness=1e12,
                tangential_stiffness=5e11,
                friction=0.5,
                timestep=1e-8,
                initial_solid_fraction=0.05,
                diagnostic_interval=1,
                snapshot_interval=100,
                stages=[dict(kind="hold", steps=2)],
            ),
        ),
    )
    return library, receipt, cache, case


def test_resolution_and_dry_run_have_no_device_dependency(inputs, capsys):
    library, _, cache, case = inputs
    before = set(sys.modules)
    resolved = resolve_case("FIXTURE", case, cache, {"particles": 64, "seed": 9})
    assert resolved["experiment"]["particles"] == 64
    assert resolved["experiment"]["seed"] == 9
    assert resolved["material"]["shape_sha256"] == sha256(library)
    assert resolved["budget"]["maximum_steps"] == 2
    assert resolved["budget"]["wall_time_estimate_s"] is None
    assert read_json(case)["experiment"]["particles"] == 27
    assert (
        main(
            [
                "case",
                "run",
                "fixture",
                "--case",
                str(case),
                "--cache",
                str(cache),
                "--dry-run",
            ]
        )
        == 0
    )
    assert '"scientific_admission": false' in capsys.readouterr().out
    assert "sand_atlas_dem.backend" not in set(sys.modules) - before


def test_hash_mismatch_and_no_overwrite(inputs):
    library, receipt, cache, _ = inputs
    with pytest.raises(FileExistsError):
        register_material(receipt, cache)
    with library.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_material("fixture", cache)


def test_ambiguous_alias_and_canonical_priority(inputs):
    _, _, cache, _ = inputs
    other = read_json(cache / "analytical-fixture.json")
    other["id"] = "second-fixture"
    other["aliases"] += ["analytical-fixture"]
    write_json(cache / "second-fixture.json", other)
    with pytest.raises(ValueError, match="found 2"):
        resolve_material("fixture", cache)
    assert resolve_material("analytical-fixture", cache)["id"] == "analytical-fixture"


def test_required_physics_unknown_fields_and_inapplicable_overrides(inputs):
    _, _, cache, case = inputs
    with pytest.raises(ValueError, match="no applicable"):
        resolve_case("fixture", case, cache, {"pressure": 1000})
    value = read_json(case)
    del value["experiment"]["friction"]
    write_json(case, value)
    with pytest.raises(ValueError, match="Explicit case parameters"):
        resolve_case("fixture", case, cache)
    value["experiment"]["friction"] = 0.5
    value["experiment"]["typo"] = 1
    write_json(case, value)
    with pytest.raises(TypeError, match="typo"):
        resolve_case("fixture", case, cache)


def test_servo_overrides_preserve_source_case(inputs):
    _, _, cache, case = inputs
    value = read_json(case)
    servo = dict(
        kind="isotropic", steps=2, pressure=1000, gain=1, filter_time=1e-5, max_rate=1
    )
    value["experiment"]["stages"] = [servo, dict(servo, kind="triaxial", rate=0.1)]
    write_json(case, value)
    result = resolve_case(
        "fixture", case, cache, {"pressure": 2000, "rate": 0.2, "stop_strain": 0.01}
    )
    stages = result["experiment"]["stages"]
    assert [s["pressure"] for s in stages] == [2000, 2000]
    assert stages[1]["rate"] == 0.2 and stages[1]["stop_strain"] == 0.01
    assert read_json(case) == value


@pytest.mark.metal
@pytest.mark.skipif(
    os.environ.get("SAND_DEM_METAL") != "1", reason="Explicit Metal execution required"
)
def test_named_case_execution_export_and_failed_preparation(inputs, tmp_path):
    _, _, cache, case = inputs
    resolved = resolve_case("fixture", case, cache)
    result = run_case(resolved, tmp_path / "complete")
    assert result["status"] == "complete"
    bundle = read_json(tmp_path / "complete/exports/atlas-comparison.json")
    assert bundle["run_status"] == "complete" and bundle["kind"] == "synthetic-dem"
    assert not bundle["scientific_admission"]
    resolved["experiment"]["stages"] = [
        dict(
            kind="isotropic",
            steps=2,
            pressure=1000,
            gain=1,
            filter_time=1e-5,
            max_rate=1,
            converged_samples=100,
        )
    ]
    result = run_case(resolved, tmp_path / "failed")
    assert result["status"] == "failed"
    bundle = read_json(tmp_path / "failed/exports/atlas-comparison.json")
    assert bundle["run_status"] == "failed"
    assert all(row["stage"] == 0 for row in bundle["rows"])


def test_initialization_failure_retains_resolved_case(inputs, tmp_path, monkeypatch):
    from types import SimpleNamespace

    _, _, cache, case = inputs
    resolved = resolve_case("fixture", case, cache)

    def fail(spec):
        raise RuntimeError("device unavailable")

    monkeypatch.setitem(
        sys.modules,
        "sand_atlas_dem.simulation",
        SimpleNamespace(Simulation=fail, export_run=None),
    )
    output = tmp_path / "init-failed"
    with pytest.raises(RuntimeError, match="device unavailable"):
        run_case(resolved, output)
    assert read_json(output / "resolved-case.json") == resolved
    assert read_json(output / "result.json")["status"] == "failed"
    with pytest.raises(FileExistsError):
        run_case(resolved, output)
