"""Mixed asset levels retain fixed-reference and prospective source bindings."""

import importlib.util
from pathlib import Path

import pytest

from sand_atlas_dem.util import sha256, write_json

source = Path(__file__).resolve().parents[1] / "examples/mesh_quadrature_study.py"
spec = importlib.util.spec_from_file_location("mixed_quadrature_study", source)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_fixed_and_mixed_levels_are_explicit():
    assert study.subdivision_for(dict(subdivisions=2), "any") == 2
    item = dict(subdivisions_by_shape={"a": 2, "b": 4})
    assert study.subdivision_for(item, "a") == 2
    assert study.subdivision_for(item, "b") == 4
    with pytest.raises(KeyError):
        study.subdivision_for(item, "unlisted")


@pytest.mark.parametrize(
    "item",
    [
        {},
        {"subdivisions": 2, "subdivisions_by_shape": {"a": 2}},
        {"subdivisions": True},
        {"subdivisions": 5},
        {"subdivisions_by_shape": {"a": -1}},
    ],
)
def test_ambiguous_or_unsupported_level_policy_rejects(item):
    with pytest.raises(ValueError):
        study.subdivision_for(item, "a")


def test_adaptive_reference_checks_archived_source_and_protocol(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, dict(inside_policy="analytical-fixture"))
    source_path = tmp_path / "source_snapshot/fixture.py"
    source_path.parent.mkdir()
    source_path.write_text("# Own synthetic reference fixture\n")
    freeze_path = tmp_path / "prospective-freeze.json"
    write_json(
        freeze_path,
        dict(
            protocol_sha256=sha256(protocol_path),
            source_hashes={source_path.name: sha256(source_path)},
        ),
    )
    policy = dict(
        reference_protocol=str(protocol_path),
        reference_protocol_sha256=sha256(protocol_path),
        reference_freeze_sha256=sha256(freeze_path),
    )
    reference = dict(source_unchanged=True, reference_refinement_passed=True)
    assert study.reference_protocol(
        reference, tmp_path / "report.json", policy
    ) == dict(inside_policy="analytical-fixture")
    source_path.write_text("changed\n")
    with pytest.raises(ValueError, match="Archived reference source"):
        study.reference_protocol(reference, tmp_path / "report.json", policy)


def test_legacy_embedded_protocol_still_supported():
    embedded = dict(nearest_point_policy="strict-fp64-minimum-squared-distance")
    assert study.reference_protocol(dict(protocol=embedded), None, {}) == embedded
