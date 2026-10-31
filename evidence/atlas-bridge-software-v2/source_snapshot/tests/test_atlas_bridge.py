"""Independent algebraic mapping and optional real Atlas Python/browser consumers."""

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from sand_atlas_dem.atlas_bridge import export_replay, project
from sand_atlas_dem.geometry import ShapeLibrary, sphere
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def analytic_case():
    """Own analytic elastic fixture; these are not GPU DEM observations."""
    servo = dict(
        kind="isotropic",
        steps=100,
        pressure=10000,
        gain=1,
        filter_time=1e-5,
        max_rate=1,
        converged_samples=100,
    )
    spec = Experiment(stages=[servo, dict(servo, kind="triaxial", steps=4, rate=0.1)])
    manifest = dict(
        format="sand-dem-run/1",
        kind="synthetic-dem",
        status="complete",
        stage_index=2,
        experiment=spec.to_dict(),
        units=dict(length="m", stress="Pa", time="s", strain="1"),
        conventions=dict(
            stress="compression_positive",
            tensor_order="xx,xy,xz;yx,yy,yz;zx,zy,zz",
            stress_measure="contact_virial_plus_peculiar_kinetic",
        ),
        source_hashes={"fixture": sha256(__file__)},
        shader_hashes={},
    )
    rows = []
    initial_cell = np.array([0.002, 0.003, 0.004])
    for i in range(1, 101):
        rows.append(
            dict(
                stage=0,
                step=i,
                elapsed_time=i * spec.timestep,
                cell=initial_cell.tolist(),
                stress=(10000 * np.eye(3)).tolist(),
                void_ratio=0.7,
                kinetic_energy=0,
                elastic_energy_proxy=1e-4,
                unbalanced_force=0,
                q=0,
            )
        )
    for i, (axial, radial_stress) in enumerate(
        [(1e-4, 11000), (2e-4, 12000), (1e-4, 10000), (1e-4, 10000)], 101
    ):
        axial_stress = 10000 + 30e6 * axial + 2 * 0.25 * (radial_stress - 10000)
        radial = (
            (1 - 0.25) * (radial_stress - 10000) - 0.25 * (axial_stress - 10000)
        ) / 30e6
        ratios = 1 - np.array([radial, radial, axial])
        rows.append(
            dict(
                stage=1,
                step=i,
                elapsed_time=i * spec.timestep,
                cell=(initial_cell * ratios).tolist(),
                stress=np.diag([radial_stress, radial_stress, axial_stress]).tolist(),
                void_ratio=1.7 * float(np.prod(ratios)) - 1,
                kinetic_energy=0,
                elastic_energy_proxy=1e-4,
                unbalanced_force=0,
                q=abs(axial_stress - radial_stress),
            )
        )
    prepared = dict(
        stage_index=1, stage_steps=0, step=100, initial_cell=initial_cell.tolist()
    )
    policy = read_json(ROOT / "cases/atlas-projection.json")
    return manifest, rows, prepared, policy


def test_algebra_prepared_origin_signed_q_and_tensor_mapping():
    manifest, rows, prepared, policy = analytic_case()
    rows[0]["cell"] = [
        0.02,
        0.03,
        0.04,
    ]  # Initial assembly must never become the loading origin.
    rows[100]["stress"] = [[12000, 30, 10], [50, 14000, 20], [40, 60, 8000]]
    source = copy.deepcopy(rows)
    result = project(manifest, rows, prepared, policy)
    assert len(result["rows"]) == 5
    assert result["rows"][0]["axial_strain"] == 0
    assert result["rows"][0]["elapsed_time"] == 0
    assert result["rows"][1]["axial_strain"] == pytest.approx(1e-4)
    assert result["rows"][1]["q"] == -5000
    assert result["rows"][1]["p"] == pytest.approx(34000 / 3)
    detail = result["dem_observations"][1]
    np.testing.assert_array_equal(
        detail["stress_tensor_atlas"],
        [[8000, 40, 60], [10, 12000, 30], [20, 50, 14000]],
    )
    assert detail["q_j2"] > 0 and detail["radial_stress_difference_pa"] == -2000
    assert detail["stress_antisymmetry_norm_pa"] > 0
    assert rows == source and result["kind"] == "synthetic-dem"
    strains = result["dem_observations"][1]["engineering_strain_atlas"]
    assert result["rows"][1]["volumetric_strain"] == pytest.approx(sum(strains))
    assert detail["finite_volume_strain"] != result["rows"][1]["volumetric_strain"]


def test_duplicate_strains_and_reversal_remain_ordered():
    result = project(*analytic_case())
    strains = [r["axial_strain"] for r in result["rows"]]
    assert strains[2] > strains[3] and strains[3] == strains[4]
    assert [r["source_step"] for r in result["dem_observations"]] == [
        100,
        101,
        102,
        103,
        104,
    ]


@pytest.mark.parametrize(
    "failure",
    [
        "status",
        "units",
        "preparation",
        "origin",
        "bound",
        "declaration",
        "time",
        "stages",
    ],
)
def test_unsupported_or_unaccepted_cases_fail(failure):
    m, r, p, policy = analytic_case()
    if failure == "status":
        m["status"] = "failed"
    elif failure == "units":
        m["units"]["stress"] = "kPa"
    elif failure == "preparation":
        r[50]["kinetic_energy"] = 1e-6
    elif failure == "origin":
        p["step"] = 99
    elif failure == "bound":
        policy["maximum_component_strain"] = 1e-5
    elif failure == "declaration":
        policy["effective_stress"] = "assumed"
    elif failure == "time":
        r[101]["elapsed_time"] = float("nan")
    elif failure == "stages":
        m["experiment"]["stages"][1]["kind"] = "strain"
    with pytest.raises(ValueError):
        project(m, r, p, policy)


@pytest.fixture
def run_fixture(tmp_path):
    """Format-correct analytic converter fixture, not an accepted DEM specimen."""
    m, rows, prepared, policy = analytic_case()
    run = tmp_path / "analytic-run"
    run.mkdir()
    library = tmp_path / "sphere.h5"
    ShapeLibrary([sphere(resolution=8, samples=16)]).save(library)
    m["experiment"]["shape_library"] = str(library)
    m["shape_sha256"] = sha256(library)
    write_json(
        run / "history.json",
        dict(format="sand-dem-history/1", kind="synthetic-dem", rows=rows),
    )
    with h5py.File(run / "prepared-stage-0.h5", "w") as f:
        f.attrs.update(
            format="sand-dem-checkpoint/3",
            shape_sha256=m["shape_sha256"],
            stage_index=1,
            stage_steps=0,
            step=100,
            initial_cell=json.dumps(prepared["initial_cell"]),
        )
    m["artifacts"] = {p.name: sha256(p) for p in run.iterdir()}
    write_json(run / "manifest.json", m)
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)
    return run, policy_path


def test_source_hash_mismatch_rejects_before_consumer(run_fixture, tmp_path):
    run, policy = run_fixture
    with (run / "history.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="artifact or hash"):
        export_replay(
            run,
            tmp_path / "out",
            policy_path=policy,
            configuration=ROOT / "cases/atlas-elastic-replay.json",
            atlas_root=tmp_path / "absent",
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.skipif(
    not os.environ.get("SAND_DEM_ATLAS_ROOT"), reason="Explicit Atlas checkout required"
)
def test_actual_atlas_python_and_browser_request_consumers(run_fixture, tmp_path):
    run, policy = run_fixture
    atlas = Path(os.environ["SAND_DEM_ATLAS_ROOT"])
    output = tmp_path / "handoff"
    result = export_replay(
        run,
        output,
        policy_path=policy,
        configuration=ROOT / "cases/atlas-elastic-replay.json",
        atlas_root=atlas,
    )
    assert result["status"] == "complete" and result["observation_count"] == 5
    request = read_json(output / "replay-request.json")
    provenance = read_json(output / "replay-request.json.provenance.json")
    assert provenance["source"]["evidence_kind"] == "synthetic-dem"
    assert request["path"]["knots"][0]["targets"] == [0, 10000, 10000, 0, 0, 0]
    for knot, source in zip(
        request["path"]["knots"], read_json(output / "virtual-test.json")["rows"]
    ):
        assert knot["targets"][0] == pytest.approx(source["axial_strain"])
        assert knot["targets"][1] == pytest.approx(source["radial_effective_stress"])
    assert shutil.which("node"), "Node required for actual browser-contract validation"
    script = """import {readFileSync} from 'node:fs';import {pathToFileURL} from 'node:url';
    const {parseRecordedJSON,validateRecordedRun}=await import(pathToFileURL(process.argv[1]));
    validateRecordedRun(parseRecordedJSON(readFileSync(process.argv[2],'utf8')));"""
    subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            script,
            str(atlas / "web/recorded.mjs"),
            str(output / "replay-request.json"),
        ],
        check=True,
    )
    with pytest.raises(FileExistsError):
        export_replay(
            run,
            output,
            policy_path=policy,
            configuration=ROOT / "cases/atlas-elastic-replay.json",
            atlas_root=atlas,
        )
    # An invalid model parameter request must retain the converter's rejection.
    bad = read_json(ROOT / "cases/atlas-elastic-replay.json")
    bad["parameters"]["E"] = -1
    bad_path = tmp_path / "bad.json"
    write_json(bad_path, bad)
    with pytest.raises(subprocess.CalledProcessError):
        export_replay(
            run,
            tmp_path / "rejected",
            policy_path=policy,
            configuration=bad_path,
            atlas_root=atlas,
        )
    assert read_json(tmp_path / "rejected/handoff.json")["status"] == "failed"
    assert (tmp_path / "rejected/converter.log").stat().st_size > 0


def test_snapshot_rejects_duplicate_declarations(run_fixture, tmp_path):
    run, policy = run_fixture
    policy.write_text(
        policy.read_text().replace(
            '"maximum_component_strain": 0.01',
            '"maximum_component_strain": 0.01, "maximum_component_strain": 0.2',
        )
    )
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        export_replay(
            run,
            tmp_path / "out",
            policy_path=policy,
            configuration=ROOT / "cases/atlas-elastic-replay.json",
            atlas_root=tmp_path / "absent",
        )


def test_observation_limit_rejects_without_downsampling():
    m, rows, p, policy = analytic_case()
    prep = rows[:100]
    template = rows[100]
    loading = []
    for i in range(5001):
        r = copy.deepcopy(template)
        r["step"] = 101 + i
        r["elapsed_time"] = (101 + i) * 1e-7
        loading.append(r)
    with pytest.raises(ValueError, match="no automatic downsampling"):
        project(m, prep + loading, p, policy)


def test_case_freezes_handoff_before_execution(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from sand_atlas_dem import cases
    from sand_atlas_dem.cli import main

    library = tmp_path / "shape.h5"
    ShapeLibrary([sphere(resolution=8, samples=16)]).save(library)
    receipt = tmp_path / "receipt.json"
    write_json(
        receipt,
        dict(
            format="sand-dem-material/1",
            id="owned-fixture",
            aliases=["fixture"],
            shape_library=str(library),
            shape_sha256=sha256(library),
            source={"kind": "analytic fixture"},
            scale_evidence="Own analytic SI fixture",
        ),
    )
    cache = tmp_path / "cache"
    cases.register_material(receipt, cache)
    case = tmp_path / "case.json"
    write_json(
        case,
        dict(
            format="sand-dem-case/1",
            experiment=dict(
                density=2650,
                normal_stiffness=1e12,
                tangential_stiffness=5e11,
                friction=0.5,
                timestep=1e-8,
                stages=[dict(kind="hold", steps=2)],
            ),
        ),
    )
    atlas = tmp_path / "atlas"
    (atlas / "atlas").mkdir(parents=True)
    (atlas / "atlas/experiment_replay.py").write_text("# consumer existence fixture")
    policy = tmp_path / "policy.json"
    shutil.copy2(ROOT / "cases/atlas-projection.json", policy)
    config = tmp_path / "config.json"
    shutil.copy2(ROOT / "cases/atlas-elastic-replay.json", config)
    original = read_json(policy)
    original_run = cases.run_case
    captured = {}

    def fake_run(resolved, output, max_chunks):
        assert resolved["atlas_handoff"]["policy"] == original
        policy.write_text(
            "{}"
        )  # A changed external policy cannot change the resolved case.
        assert resolved["atlas_handoff"]["policy"] == original
        captured["resolved"] = resolved
        return dict(status="paused")

    monkeypatch.setattr(cases, "run_case", fake_run)
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
                "--output",
                str(tmp_path / "out"),
                "--atlas-root",
                str(atlas),
                "--atlas-policy",
                str(policy),
                "--atlas-config",
                str(config),
            ]
        )
        == 0
    )
    output = tmp_path / "initialization-check"

    def initializer(spec):
        assert read_json(output / "atlas-policy.json") == original
        assert read_json(output / "atlas-configuration.json") == read_json(config)
        raise RuntimeError("verified frozen inputs before device initialization")

    monkeypatch.setitem(
        sys.modules,
        "sand_atlas_dem.simulation",
        SimpleNamespace(Simulation=initializer, export_run=None),
    )
    with pytest.raises(RuntimeError, match="verified frozen inputs"):
        original_run(captured["resolved"], output)
