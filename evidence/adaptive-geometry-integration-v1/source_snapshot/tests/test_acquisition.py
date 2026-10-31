"""Source pinning, exclusion decisions and recoverable preparation on own meshes."""

import csv
import hashlib
import io
import zipfile

import pytest
import trimesh

from sand_atlas_dem import acquisition
from sand_atlas_dem.cases import resolve_material
from sand_atlas_dem.cli import main
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.util import read_json, sha256, write_json


def test_pinned_download_and_cache_corruption(tmp_path, monkeypatch):
    payload = b"owned source fixture"
    source = dict(
        filename="source.zip",
        url="https://example.invalid/source",
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **k: io.BytesIO(payload))
    path = acquisition.fetch_pinned(source, tmp_path)
    assert path.read_bytes() == payload
    monkeypatch.setattr(
        acquisition, "urlopen", lambda *a, **k: pytest.fail("unexpected network")
    )
    assert acquisition.fetch_pinned(source, tmp_path) == path
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Cached source hash mismatch"):
        acquisition.fetch_pinned(source, tmp_path)


@pytest.mark.parametrize("failure", ["hash", "budget", "interruption"])
def test_failed_download_is_not_published(tmp_path, monkeypatch, failure):
    class Interrupted(io.BytesIO):
        def read(self, *args):
            raise OSError("connection interrupted")

    stream = Interrupted() if failure == "interruption" else io.BytesIO(b"bad")
    source = dict(
        filename="source.zip",
        url="https://example.invalid/source",
        sha256="0" * 64,
        bytes=1 if failure == "budget" else 20,
    )
    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **k: stream)
    with pytest.raises((ValueError, OSError)):
        acquisition.fetch_pinned(source, tmp_path)
    assert not list(tmp_path.iterdir())


def test_reviewed_recipe_has_complete_unique_crosswalk():
    recipe, _ = acquisition.source_recipe("HOSTUN")
    assert len(recipe["members"]) == 54
    assert sum(m["accepted"] for m in recipe["members"]) == 53
    assert len({m["descriptor_id"] for m in recipe["members"]}) == 54
    assert [m["name"] for m in recipe["members"] if not m["accepted"]] == [
        "particle_00019.stl"
    ]
    with pytest.raises(ValueError, match="No reviewed source/scale recipe"):
        acquisition.source_recipe("unreviewed-sand")


@pytest.fixture
def fixture_recipe(tmp_path, monkeypatch):
    sources = tmp_path / "sources"
    sources.mkdir()
    good = trimesh.creation.box([10, 10, 10])
    a = trimesh.creation.box([3, 3, 3])
    b = a.copy()
    b.apply_translation([10, 0, 0])
    bad = trimesh.util.concatenate([a, b])
    archive = sources / "source.zip"
    members = []
    with zipfile.ZipFile(archive, "w") as z:
        for i, mesh in enumerate([good, bad]):
            data = mesh.export(file_type="stl")
            name = f"grain-{i}.stl"
            z.writestr(name, data)
            members.append(
                dict(
                    name=name,
                    sha256=hashlib.sha256(data).hexdigest(),
                    descriptor_id=str(i),
                    accepted=i == 0,
                )
            )
    table = sources / "source.csv"
    with table.open("w") as stream:
        writer = csv.DictWriter(stream, ["Particle ID", "Volume (µm³)"])
        writer.writeheader()
        for i, mesh in enumerate([good, bad]):
            writer.writerow({"Particle ID": str(i), "Volume (µm³)": mesh.volume})
    recipe = dict(
        format="sand-dem-source-recipe/1",
        id="owned-fixture",
        aliases=["fixture"],
        scale_evidence="Owned synthetic coordinates explicitly defined as micrometres.",
        coordinate_unit="um",
        repairs=[],
        members=members,
    )
    for key, path in [("archive", archive), ("descriptors", table)]:
        recipe[key] = dict(
            url="https://example.invalid/" + path.name,
            filename=path.name,
            sha256=sha256(path),
            bytes=path.stat().st_size,
        )
    path = tmp_path / "recipe.json"
    write_json(path, recipe)
    monkeypatch.setattr(acquisition, "source_recipe", lambda name: (recipe, path))
    monkeypatch.setattr(
        acquisition, "urlopen", lambda *a, **k: pytest.fail("unexpected network")
    )
    return recipe, sources, tmp_path / "cache"


def test_prepare_register_reuse_and_profile_conflict(fixture_recipe):
    recipe, sources, cache = fixture_recipe
    record = acquisition.prepare_material(
        "fixture", cache, resolution=8, samples=16, workers=1, source_directory=sources
    )
    assert resolve_material("fixture", cache) == record
    report = read_json(record["source"]["preparation_report"])
    assert report["status"] == "complete" and (
        report["accepted"],
        report["excluded"],
    ) == (1, 1)
    shape = ShapeLibrary.load(record["shape_library"]).shapes[0]
    assert shape.volume == pytest.approx(1000e-18)
    assert shape.provenance["repairs"] == []
    assert shape.provenance["descriptor_id"] == recipe["members"][0]["descriptor_id"]
    assert (
        acquisition.prepare_material(
            "fixture", cache, resolution=8, samples=16, workers=1
        )
        == record
    )
    with pytest.raises(ValueError, match="another preparation profile"):
        acquisition.prepare_material(
            "fixture", cache, resolution=9, samples=16, workers=1
        )
    report_path = record["source"]["preparation_report"]
    write_json(report_path, {})
    with pytest.raises(ValueError, match="Preparation report hash mismatch"):
        acquisition.prepare_material(
            "fixture", cache, resolution=8, samples=16, workers=1
        )


def test_changed_inventory_retains_failure_and_does_not_register(fixture_recipe):
    recipe, sources, cache = fixture_recipe
    recipe["members"] = recipe["members"][:1]
    # Keep descriptor completeness so inventory checking is reached.
    with (sources / "source.csv").open("w") as stream:
        stream.write("Particle ID,Volume (µm³)\n0,1000\n")
    recipe["descriptors"]["sha256"] = sha256(sources / "source.csv")
    with pytest.raises(ValueError, match="Archive member inventory"):
        acquisition.prepare_material(
            "fixture",
            cache,
            resolution=8,
            samples=16,
            workers=1,
            source_directory=sources,
        )
    assert not (cache / "owned-fixture.json").exists()
    reports = list(cache.glob("private/*/*/attempt-*.json"))
    assert len(reports) == 1 and read_json(reports[0])["status"] == "failed"
    assert not list(cache.rglob("preparation.lock"))


def test_dry_run_and_invalid_case_never_acquire(tmp_path, monkeypatch):
    case = tmp_path / "case.json"
    value = dict(
        format="sand-dem-case/1",
        experiment=dict(
            density=2650,
            normal_stiffness=1e12,
            tangential_stiffness=5e11,
            friction=0.5,
            timestep=1e-9,
            stages=[dict(kind="hold", steps=2)],
        ),
    )
    write_json(case, value)
    monkeypatch.setattr(
        acquisition,
        "prepare_material",
        lambda *a, **k: pytest.fail("unexpected acquisition"),
    )
    args = [
        "case",
        "run",
        "hostun",
        "--case",
        str(case),
        "--cache",
        str(tmp_path / "cache"),
    ]
    assert main(args + ["--dry-run"]) == 1
    value["experiment"]["typo"] = 1
    write_json(case, value)
    assert main(args + ["--output", str(tmp_path / "run")]) == 1


def test_case_first_use_prepares_then_resolves(fixture_recipe, tmp_path, monkeypatch):
    from sand_atlas_dem import cases

    _, sources, cache = fixture_recipe
    original = acquisition.prepare_material
    calls = []

    def prepare(name, directory):
        calls.append(name)
        return original(
            name,
            directory,
            resolution=8,
            samples=16,
            workers=1,
            source_directory=sources,
        )

    def run(resolved, output, max_chunks):
        assert resolved["material"]["id"] == "owned-fixture"
        assert (
            sha256(resolved["experiment"]["shape_library"])
            == resolved["material"]["shape_sha256"]
        )
        return dict(status="complete")

    monkeypatch.setattr(acquisition, "prepare_material", prepare)
    monkeypatch.setattr(cases, "run_case", run)
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
                timestep=1e-9,
                stages=[dict(kind="hold", steps=2)],
            ),
        ),
    )
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
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    assert calls == ["fixture"]


def test_failed_registration_preserves_completed_receipt_for_retry(
    fixture_recipe, monkeypatch
):
    _, sources, cache = fixture_recipe
    original = acquisition.register_material

    def fail(*args):
        raise OSError("registry write unavailable")

    monkeypatch.setattr(acquisition, "register_material", fail)
    with pytest.raises(OSError, match="registry write unavailable"):
        acquisition.prepare_material(
            "fixture",
            cache,
            resolution=8,
            samples=16,
            workers=1,
            source_directory=sources,
        )
    receipt_path = next(cache.glob("private/*/*/material.json"))
    receipt = read_json(receipt_path)
    completed = receipt["source"]["preparation_report"]
    assert read_json(completed)["status"] == "complete"
    assert sha256(completed) == receipt["source"]["preparation_report_sha256"]
    assert len(list(receipt_path.parent.glob("*.registration-failure.json"))) == 1
    monkeypatch.setattr(acquisition, "register_material", original)
    record = acquisition.prepare_material(
        "fixture", cache, resolution=8, samples=16, workers=1, source_directory=sources
    )
    assert record == receipt
    assert sha256(completed) == receipt["source"]["preparation_report_sha256"]
