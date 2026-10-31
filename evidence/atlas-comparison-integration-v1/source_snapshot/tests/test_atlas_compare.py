"""Compare real captured native responses, preserving source order and stress meaning."""

import copy
from pathlib import Path

import pytest

from sand_atlas_dem.atlas_compare import (
    canonical_hash,
    comparison_model,
    comparison_rows,
)
from sand_atlas_dem.util import read_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def captured():
    evidence = ROOT / "evidence/atlas-native-replay-v2"
    virtual = read_json(evidence / "virtual-test.json")
    request = read_json(evidence / "replay-request.json")
    response = read_json(evidence / "native-response.json")
    model = read_json(evidence / "source_snapshot/atlas/models/linear-elastic.json")
    return virtual, request, response, model


def test_actual_native_response_retains_reversal_duplicate_and_signed_stress(captured):
    virtual, request, response, model = captured
    rows = comparison_rows(
        virtual, request, response, model, response["catalogue_hash"]
    )
    assert [r["source_row_index"] for r in rows] == [0, 1, 2, 3, 4]
    assert [r["synthetic_dem"]["q"] for r in rows] == [0, 2500, 5000, 3000, 3000]
    assert [r["continuum"]["q"] for r in rows] == pytest.approx(
        [0, 2500, 5000, 3000, 3000], abs=1e-7
    )
    assert rows[2]["continuum"]["axial_strain"] > rows[3]["continuum"]["axial_strain"]
    assert rows[3]["continuum"] == rows[4]["continuum"]
    assert rows[3]["elapsed_time_s"] < rows[4]["elapsed_time_s"]
    for row in rows:
        assert row["continuum"]["volumetric_strain"] == pytest.approx(
            row["synthetic_dem"]["volumetric_strain"], abs=1e-12
        )
    # Signed axial-minus-radial stress must remain negative on extension.
    response["points"][1]["state"][0] = 9000
    rows = comparison_rows(
        virtual, request, response, model, response["catalogue_hash"]
    )
    assert rows[1]["continuum"]["q"] == -2000


@pytest.mark.parametrize(
    "change",
    [
        "catalogue",
        "coordinate",
        "truncated",
        "failed",
        "parameters",
        "nan",
        "layout",
        "request",
    ],
)
def test_native_binding_and_complete_inventory_reject_incompatible_results(
    captured, change
):
    virtual, request, response, model = captured
    identity = response["catalogue_hash"]
    if change == "catalogue":
        response["catalogue_hash"] = "0" * 64
    if change == "coordinate":
        response["points"][1]["coordinate"] = 2
    if change == "truncated":
        response["points"].pop()
    if change == "failed":
        response["ok"] = False
    if change == "parameters":
        response["parameters"][1] *= 2
    if change == "nan":
        response["points"][1]["state"][0] = float("nan")
    if change == "layout":
        response["points"][1]["state"].append(0)
    if change == "request":
        response["request_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        comparison_rows(virtual, request, response, model, identity)


def test_catalogue_binding_uses_the_declared_incremental_contract(captured):
    model = captured[-1]
    catalogue = dict(
        schema_version="0.1", models=[model], sha256=canonical_hash([model])
    )
    selected, binding = comparison_model(catalogue, model["id"])
    assert selected is model
    assert binding["name"] == "sigma" and binding["scope"] == "effective"
    # Reject a model whose state order cannot implement its stress binding.
    broken = copy.deepcopy(model)
    broken["state"].reverse()
    bad = dict(catalogue, models=[broken], sha256=canonical_hash([broken]))
    with pytest.raises(ValueError, match="stress binding"):
        comparison_model(bad, model["id"])
    # A forged catalogue hash cannot bind any binary/response.
    with pytest.raises(ValueError, match="catalogue identity"):
        comparison_model(dict(catalogue, sha256="0" * 64), model["id"])


@pytest.mark.parametrize("profile", ["physical_time", "total", "missing"])
def test_unsupported_explicit_profiles_do_not_infer_stress(captured, profile):
    model = copy.deepcopy(captured[-1])
    model["schema_version"] = "0.3"
    model.pop("integration")
    model["formulation"] = dict(
        time_dependence="strain_path",
        equation_form="incremental_residual",
        kinematics="small_strain_3d",
        frame="fixed_spatial",
        objective_rate="none",
        tensor_representation="mandel",
        stress_sign="compression_positive",
        strain_sign="compression_positive",
        stress=dict(source="state", name="sigma", measure="cauchy", scope="effective"),
    )
    if profile == "physical_time":
        model["formulation"]["time_dependence"] = "physical_time"
    if profile == "total":
        model["formulation"]["stress"]["scope"] = "total"
    if profile == "missing":
        model["formulation"].pop("stress")
    catalogue = dict(
        schema_version="0.1", models=[model], sha256=canonical_hash([model])
    )
    with pytest.raises(ValueError, match="unsupported"):
        comparison_model(catalogue, model["id"])


def test_nonzero_continuum_strain_origin_is_not_counted_as_dem_loading(captured):
    virtual, request, response, model = captured
    shift = [0.01, 0.02, 0.02, 0, 0, 0]
    request["path"]["initial_strain"] = shift[:]
    for knot in request["path"]["knots"]:
        knot["targets"][0] += shift[0]
    response["specification"] = copy.deepcopy(request)
    response["request_sha256"] = canonical_hash(request)
    response["initial_strain"] = shift[:]
    for point in response["points"]:
        point["strain"] = [x + y for x, y in zip(point["strain"], shift)]
    rows = comparison_rows(
        virtual, request, response, model, response["catalogue_hash"]
    )
    assert rows[0]["continuum"]["volumetric_strain"] == 0
    assert rows[2]["continuum"]["axial_strain"] == pytest.approx(0.0002, abs=1e-15)
    assert rows[2]["continuum"]["volumetric_strain"] == pytest.approx(
        0.0001833333333333, abs=1e-15
    )


@pytest.fixture
def native_settings():
    import os

    names = ["SAND_DEM_ATLAS_ROOT", "SAND_DEM_ATLAS_SOLVER", "SAND_DEM_ATLAS_CATALOGUE"]
    if not all(os.environ.get(name) for name in names):
        pytest.skip(
            "Explicit Atlas checkout, existing native solver and generated catalogue required"
        )
    return [Path(os.environ[name]) for name in names]


def test_actual_native_comparison_cli_retains_source_and_binary_identity(
    tmp_path, native_settings
):
    import os
    import subprocess
    import sys

    atlas, solver, catalogue = native_settings
    output = tmp_path / "comparison"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sand_atlas_dem.cli",
            "atlas",
            "compare",
            str(ROOT / "evidence/atlas-bridge-software-v3/handoff"),
            "--atlas-root",
            str(atlas),
            "--solver",
            str(solver),
            "--catalogue",
            str(catalogue),
            "--output",
            str(output),
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = read_json(output / "report.json")
    comparison = read_json(output / "comparison.json")
    assert report["status"] == "complete" and report["source_unchanged"]
    assert not report["scientific_admission"]
    assert report["observation_count"] == 5
    assert comparison["maximum_absolute_difference"]["q_Pa"] < 1e-7
    assert len(comparison["rows"]) == 5


def test_native_catalogue_mismatch_rejects_before_replay(tmp_path, native_settings):
    from sand_atlas_dem.atlas_compare import replay_comparison
    from sand_atlas_dem.util import write_json

    atlas, solver, catalogue = native_settings
    altered = read_json(catalogue)
    other = next(m for m in altered["models"] if m["id"] != "linear-elastic")
    other["description"] += (
        " Deliberate catalogue identity mismatch for rejection test."
    )
    altered["sha256"] = canonical_hash(altered["models"])
    path = tmp_path / "different-catalogue.json"
    write_json(path, altered)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="binary and supplied catalogue differ"):
        replay_comparison(
            ROOT / "evidence/atlas-bridge-software-v3/handoff",
            output,
            atlas_root=atlas,
            solver=solver,
            catalogue=path,
        )
    assert read_json(output / "report.json")["status"] == "failed"
    assert (output / "native-describe.json").exists()
    assert not (output / "native-response.json").exists()
