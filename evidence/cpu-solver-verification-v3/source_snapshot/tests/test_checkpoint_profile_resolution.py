"""CPU-only freezing contracts for the original checkpoint profiling workload."""

import importlib.util
import json
from pathlib import Path

import h5py
import pytest

source = Path(__file__).resolve().parents[1] / "examples/profile_checkpoint.py"
spec = importlib.util.spec_from_file_location("checkpoint_profile", source)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


@pytest.fixture
def inputs(tmp_path):
    study = tmp_path / "study"
    archive = study / "source_snapshot"
    names = [
        "__init__.py",
        "gpu.py",
        "backend.py",
        "simulation.py",
        "geometry.py",
        "spec.py",
        "benchmark.py",
        "shaders/solver.slang",
    ]
    sources = {}
    for name in names:
        key = "sand_atlas_dem/" + name
        p = archive / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("original archived source\n")
        sources[key] = profile.digest(p)
    shape = tmp_path / "shapes.h5"
    shape.write_bytes(b"original shape identity")
    profile.write(
        study / "frozen-manifest.json",
        dict(sources=sources, shape_sha256=profile.digest(shape)),
    )
    checkpoint = tmp_path / "checkpoint.h5"
    experiment = dict(
        shape_library=str(shape),
        contact_geometry="mesh",
        particles=424,
        periodic=True,
        planes=[],
        normal_viscosity=0,
        stages=[
            dict(kind="isotropic", drag=100000, max_rate=1000, contact_max_rate=50)
        ],
    )
    with h5py.File(checkpoint, "w") as f:
        f.attrs.update(
            experiment=json.dumps(experiment),
            step=500000,
            contact_phase=True,
            stage_index=0,
            shape_sha256=profile.digest(shape),
        )
    policy = dict(
        identity="profile-fixture",
        checkpoint=str(checkpoint),
        checkpoint_sha256=profile.digest(checkpoint),
        checkpoint_step=500000,
        preparation_study=str(study),
        particles=424,
        drag_s=100000,
        rate_s=[-50, -50, -50],
        warmup_steps=16,
        profile_steps=128,
        repetitions=3,
        scope="Unaccepted first-contact profiling diagnostic only",
    )
    path = tmp_path / "policy.json"
    profile.write(path, policy)
    return path, policy, study, shape


def test_resolves_exact_original_archive_without_executing_solver(inputs, tmp_path):
    path, policy, study, shape = inputs
    output = tmp_path / "resolved"
    driver = profile.resolve(path, output)
    assert driver.read_bytes() == source.read_bytes()
    resolved = profile.read(output / "protocol.json")
    assert not resolved["scientific_admission"] and not resolved["preparation_accepted"]
    assert resolved["drag_s"] == 100000 and resolved["rate_s"] == [-50] * 3
    assert profile.digest(resolved["checkpoint"]) == policy["checkpoint_sha256"]
    assert (
        output / "source_snapshot/sand_atlas_dem/gpu.py"
    ).read_text() == "original archived source\n"
    freeze = profile.read(output / "prospective-freeze.json")
    assert not freeze["gpu_executed"]
    assert freeze["protocol_sha256"] == profile.digest(output / "protocol.json")
    assert not (output / "profile").exists()


@pytest.mark.parametrize(
    "change,error",
    [
        ("checkpoint", "checkpoint changed"),
        ("archive", "source archive changed"),
        ("shape", "shape asset changed"),
        ("rate", "rate exceeds"),
        ("workload", "Positive integer"),
    ],
)
def test_changed_or_unsupported_inputs_fail_before_output(
    inputs, tmp_path, change, error
):
    path, policy, study, shape = inputs
    if change == "checkpoint":
        with h5py.File(policy["checkpoint"], "a") as f:
            f.attrs["step"] = 500001
    elif change == "archive":
        (study / "source_snapshot/sand_atlas_dem/gpu.py").write_text("changed")
    elif change == "shape":
        shape.write_bytes(b"changed")
    elif change == "rate":
        policy["rate_s"] = [-51] * 3
        profile.write(path, policy)
    else:
        policy["profile_steps"] = True
        profile.write(path, policy)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=error):
        profile.resolve(path, output)
    assert not output.exists()


def test_rejects_source_path_escape_before_copy(inputs, tmp_path):
    path, policy, study, shape = inputs
    receipt = profile.read(study / "frozen-manifest.json")
    receipt["sources"]["../escape"] = "invalid"
    profile.write(study / "frozen-manifest.json", receipt)
    with pytest.raises(ValueError, match="Unsafe archived source path"):
        profile.resolve(path, tmp_path / "rejected")


def test_execution_rejects_changed_protocol_before_importing_solver(
    inputs, tmp_path, monkeypatch
):
    path, policy, study, shape = inputs
    output = tmp_path / "resolved"
    profile.resolve(path, output)
    p = profile.read(output / "protocol.json")
    p["drag_s"] = 0
    profile.write(output / "protocol.json", p)
    import builtins

    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert not name.startswith("sand_atlas_dem"), (
            "Solver imported before input checks"
        )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    with pytest.raises(ValueError, match="Frozen profiling protocol changed"):
        profile.execute(output / "protocol.json", output / "profile")
    assert not (output / "profile").exists()
