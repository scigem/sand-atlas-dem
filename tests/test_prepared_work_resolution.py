"""CPU-only metadata contracts; fixtures are not accepted scientific preparations."""

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import pytest

from sand_atlas_dem.util import read_json, sha256, write_json

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "work_resolver", root / "examples/resolve_prepared_work.py"
)
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)


@pytest.fixture
def fixture(tmp_path):
    preparation = tmp_path / "preparation"
    (preparation / "study").mkdir(parents=True)
    shape = tmp_path / "metadata-fixture.bin"
    shape.write_bytes(b"CPU metadata fixture only; no usable geometry")
    checkpoint = tmp_path / "metadata-checkpoint.h5"
    with h5py.File(checkpoint, "w") as f:
        f.attrs["experiment"] = json.dumps(
            dict(
                contact_geometry="mesh",
                particles=2,
                normal_viscosity=0,
                stages=[dict(pressure=10000)],
            )
        )
    frozen = preparation / "study/frozen-manifest.json"
    write_json(
        frozen,
        dict(
            status="complete",
            return_code=0,
            shape_sha256=sha256(shape),
            sources={
                "sand_atlas_dem/spec.py": sha256(root / "src/sand_atlas_dem/spec.py")
            },
        ),
    )
    audit = preparation / "acceptance-audit.json"
    write_json(
        audit,
        dict(
            passed=True,
            source_unchanged=True,
            shape_counts={"fixture": 2},
            prepared_checkpoint=str(checkpoint),
            prepared_checkpoint_sha256=sha256(checkpoint),
            shape_library=str(shape),
            shape_library_sha256=sha256(shape),
            input_hashes={str(frozen): sha256(frozen)},
        ),
    )
    write_json(preparation / "report.json", dict(passed=True, source_unchanged=True))
    policy = tmp_path / "policy.json"
    data = read_json(root / "studies/prepared-hostun424-assembly-work-v1.json")
    data.update(
        requires_preparation=str(preparation),
        particles=2,
        required_source_shapes=1,
        copies_per_shape=2,
        scope="CPU metadata unit fixture only",
    )
    write_json(policy, data)
    return policy, preparation, checkpoint, shape


def invoke(monkeypatch, policy, output):
    monkeypatch.setattr(sys, "argv", ["resolver", str(policy), "--output", str(output)])
    return resolver.main()


def test_resolution_preserves_frozen_workload_and_archives_sources(
    fixture, tmp_path, monkeypatch
):
    policy, _, _, _ = fixture
    output = tmp_path / "resolved"
    assert invoke(monkeypatch, policy, output) == 0
    original, resolved = read_json(policy), read_json(output / "protocol.json")
    for key in (
        "cases",
        "timesteps_s",
        "maximum_relative_proxy_balance_residual",
        "maximum_relative_independent_work_error",
        "maximum_relative_refinement_difference",
    ):
        assert resolved[key] == original[key]
    freeze = read_json(output / "prospective-freeze.json")
    assert freeze["protocol_sha256"] == sha256(output / "protocol.json")
    for name, digest in freeze["source_hashes"].items():
        assert sha256(output / "source_snapshot" / name) == digest
    assert not (output / "study").exists()  # Resolution never starts Metal.


@pytest.mark.parametrize(
    "mutation",
    [
        "unaccepted",
        "inventory",
        "geometry",
        "viscosity",
        "pressure",
        "changed-shape",
        "changed-manifest",
    ],
)
def test_invalid_preparation_rejects_before_output(
    fixture, tmp_path, monkeypatch, mutation
):
    policy, preparation, checkpoint, shape = fixture
    if mutation in ("unaccepted", "inventory"):
        path = preparation / "acceptance-audit.json"
        data = read_json(path)
        if mutation == "unaccepted":
            data["passed"] = False
        else:
            data["shape_counts"] = {"fixture": 1}
        write_json(path, data)
    elif mutation in ("geometry", "viscosity", "pressure"):
        with h5py.File(checkpoint, "a") as f:
            data = json.loads(f.attrs["experiment"])
            if mutation == "geometry":
                data["contact_geometry"] = "sdf"
            if mutation == "viscosity":
                data["normal_viscosity"] = 1
            if mutation == "pressure":
                data["stages"][0]["pressure"] = 5000
            f.attrs["experiment"] = json.dumps(data)
        path = preparation / "acceptance-audit.json"
        data = read_json(path)
        data["prepared_checkpoint_sha256"] = sha256(checkpoint)
        write_json(path, data)
    elif mutation == "changed-shape":
        shape.write_bytes(b"changed")
    else:
        path = preparation / "study/frozen-manifest.json"
        data = read_json(path)
        data["status"] = "running"
        write_json(path, data)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        invoke(monkeypatch, policy, output)
    assert not output.exists()
