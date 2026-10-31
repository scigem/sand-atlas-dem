"""Fixed analytical selection tables and physical-asset preservation checks."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import trimesh

from sand_atlas_dem.cli import main
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.quadrature import prepare_adaptive, select_levels
from sand_atlas_dem.util import read_json, sha256, write_json


def reference_fixture():
    rows = []
    for name in ("a", "b"):
        for depth in (0.0025, 0.01):
            # Closed synthetic force table, not a claimed DEM/reference study.
            scale = 1 if depth == 0.0025 else 2
            multipliers = [2, 1.3, 1.04, 1.01, 1]
            if name == "a" and depth == 0.01:
                multipliers[2] = 0.9
            forces = [[-scale * m, 0, 0] for m in multipliers]
            rows.append(
                dict(
                    shape=name,
                    target_sampled_depth_fraction=depth,
                    separation_m=0.001,
                    reference_finest_level=4,
                    finest_relative_change=0.01,
                    levels=[
                        dict(level=i, force_integral_m3=f) for i, f in enumerate(forces)
                    ],
                )
            )
    return dict(
        passed=True,
        source_unchanged=True,
        reference_refinement_passed=True,
        expected_cases=4,
        completed_cases=4,
        failures=[],
        rows=rows,
    ), dict(shape_ids=["a", "b"], target_sampled_depth_fractions=[0.0025, 0.01])


def test_selection_requires_every_pose_and_uses_minimum_supported_level():
    reference, protocol = reference_fixture()
    selection = select_levels(reference, protocol, ["b", "a"])
    assert [(r["shape"], r["subdivisions"]) for r in selection] == [("a", 3), ("b", 2)]
    assert selection[0]["worst_reference_relative_error"] == pytest.approx(0.01)
    assert selection[1]["worst_reference_relative_error"] == pytest.approx(0.04)
    assert all(len(r["poses"]) == 2 for r in selection)


@pytest.mark.parametrize(
    "field", ["passed", "source_unchanged", "reference_refinement_passed"]
)
def test_rejects_failed_reference_flags(field):
    reference, protocol = reference_fixture()
    reference[field] = False
    with pytest.raises(ValueError, match="passing source-stable"):
        select_levels(reference, protocol, ["a", "b"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "count"])
def test_rejects_incomplete_pose_inventory(mutation):
    reference, protocol = reference_fixture()
    if mutation == "missing":
        reference["rows"].pop()
    elif mutation == "duplicate":
        reference["rows"][-1] = copy.deepcopy(reference["rows"][0])
    elif mutation == "unknown":
        reference["rows"][-1]["shape"] = "unknown"
    else:
        reference["completed_cases"] = 3
    with pytest.raises(ValueError, match="Complete unique"):
        select_levels(reference, protocol, ["a", "b"])


def test_rejects_reference_self_comparison_and_forged_convergence_flag():
    reference, protocol = reference_fixture()
    # Finest level would have zero self-error; it cannot qualify itself.
    with pytest.raises(ValueError, match="No independently supported"):
        select_levels(reference, protocol, ["a", "b"], target=0.001)
    reference["rows"][0]["finest_relative_change"] = 0
    with pytest.raises(ValueError, match="unchanged 2%"):
        select_levels(reference, protocol, ["a", "b"])


def test_finer_independent_reference_retains_fixed_candidate_limit():
    reference, protocol = reference_fixture()
    for row in reference["rows"]:
        final = np.array(row["levels"][-1]["force_integral_m3"])
        for level, factor in enumerate([1.002, 1.0005, 1.0002, 1], 5):
            row["levels"].append(
                dict(level=level, force_integral_m3=(final * factor).tolist())
            )
        row["reference_finest_level"] = 8
        row["finest_relative_change"] = 0.0002
    selected = select_levels(reference, protocol, ["a", "b"])
    assert [r["subdivisions"] for r in selected] == [3, 2]
    reference["rows"][0]["levels"].pop(5)
    with pytest.raises(ValueError, match="Complete ordered"):
        select_levels(reference, protocol, ["a", "b"])


@pytest.fixture
def frozen_fixture(tmp_path):
    shape = from_mesh(
        trimesh.creation.box(extents=[0.001] * 3),
        shape_id="a",
        unit="m",
        resolution=8,
        samples=16,
    )
    other = replace(shape, id="b", provenance=dict(shape.provenance))
    library = tmp_path / "parent.h5"
    ShapeLibrary([shape, other]).save(library)
    reference, protocol = reference_fixture()
    protocol["library_sha256"] = sha256(library)
    report = tmp_path / "report.json"
    protocol_path = tmp_path / "protocol.json"
    write_json(report, reference)
    write_json(protocol_path, protocol)
    source = tmp_path / "source_snapshot/analytical_fixture.py"
    source.parent.mkdir()
    source.write_text("# Fixed analytical force table, not GPU simulation evidence.\n")
    write_json(
        tmp_path / "prospective-freeze.json",
        dict(
            protocol_sha256=sha256(protocol_path),
            source_hashes={source.name: sha256(source)},
        ),
    )
    return library, report, protocol_path


def test_preparation_preserves_every_physical_shape_array(frozen_fixture, tmp_path):
    library, reference, protocol = frozen_fixture
    output = tmp_path / "adaptive.h5"
    receipt = prepare_adaptive(library, reference, protocol, output)
    before, after = ShapeLibrary.load(library), ShapeLibrary.load(output)
    assert [s.provenance["quadrature_subdivisions"] for s in after.shapes] == [3, 2]
    for a, b in zip(before.shapes, after.shapes):
        for name in ("vertices", "faces", "sdf", "origin", "inertia"):
            assert np.array_equal(getattr(a, name), getattr(b, name)), name
        for name in ("id", "spacing", "radius", "volume"):
            assert getattr(a, name) == getattr(b, name), name
        level = b.provenance["quadrature_subdivisions"]
        assert len(b.samples) == len(a.faces) * 3 * 4**level
        assert b.areas.sum() == pytest.approx(6e-6, rel=1e-12)
        assert b.provenance["parent_shape_library_sha256"] == sha256(library)
        assert not b.provenance["quadrature_gpu_verified"]
    assert receipt["static_reference_target_passed"]
    assert not receipt["gpu_verified"] and not receipt["scientific_admission"]
    assert receipt["sha256"] == sha256(output)
    with pytest.raises(FileExistsError):
        prepare_adaptive(library, reference, protocol, output)


@pytest.mark.parametrize("mutation", ["protocol", "source", "library"])
def test_rejects_changed_frozen_inputs_before_output(
    frozen_fixture, tmp_path, mutation
):
    library, reference, protocol = frozen_fixture
    if mutation == "source":
        (tmp_path / "source_snapshot/analytical_fixture.py").write_text("changed\n")
    else:
        path = protocol if mutation == "protocol" else library
        with path.open("ab") as f:
            f.write(b" ")
    output = tmp_path / "rejected.h5"
    with pytest.raises(ValueError, match="freeze|snapshot|hash"):
        prepare_adaptive(library, reference, protocol, output)
    assert not output.exists()


def test_actual_cli_prepares_mixed_level_library(frozen_fixture, tmp_path, capsys):
    library, reference, protocol = frozen_fixture
    output = tmp_path / "cli.h5"
    assert (
        main(
            [
                "geometry",
                "adaptive-quadrature",
                str(library),
                str(output),
                "--reference",
                str(reference),
                "--reference-protocol",
                str(protocol),
            ]
        )
        == 0
    )
    import json

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["format"] == "sand-dem-adaptive-quadrature/1"
    assert receipt["total_samples"] == (4**3 + 4**2) * 12 * 3
    assert not receipt["gpu_verified"]
    assert read_json(protocol)["library_sha256"] == receipt["parent_sha256"]


def high_resolution_reference():
    reference, protocol = reference_fixture()
    for row in reference["rows"]:
        scale = 1 if row["target_sampled_depth_fraction"] == 0.0025 else 2
        row["levels"] = [
            dict(level=i, force_integral_m3=[-scale * m, 0, 0])
            for i, m in enumerate([2, 1.5, 1.3, 1.2, 1.1, 1.02, 1.005, 1.001, 1])
        ]
        row["reference_finest_level"] = 8
        row["finest_relative_change"] = 0.001
    return reference, protocol


def test_higher_candidate_limit_requires_explicit_opt_in():
    reference, protocol = high_resolution_reference()
    with pytest.raises(ValueError, match="No independently supported"):
        select_levels(reference, protocol, ["a", "b"])
    for limit in (5, 6):
        selected = select_levels(reference, protocol, ["a", "b"], maximum_level=limit)
        assert [r["subdivisions"] for r in selected] == [5, 5]


@pytest.mark.parametrize("limit", [1, 7, True, 4.0])
def test_invalid_candidate_limit_rejects(limit):
    reference, protocol = high_resolution_reference()
    with pytest.raises(ValueError, match="Maximum quadrature level"):
        select_levels(reference, protocol, ["a", "b"], maximum_level=limit)


def test_high_level_cli_preserves_physical_data_and_explicit_bound(
    frozen_fixture, tmp_path
):
    library, reference_path, protocol = frozen_fixture
    reference, _ = high_resolution_reference()
    write_json(reference_path, reference)
    output = tmp_path / "high.h5"
    assert (
        main(
            [
                "geometry",
                "adaptive-quadrature",
                str(library),
                str(output),
                "--reference",
                str(reference_path),
                "--reference-protocol",
                str(protocol),
                "--maximum-level",
                "5",
            ]
        )
        == 0
    )
    before, after = ShapeLibrary.load(library), ShapeLibrary.load(output)
    for a, b in zip(before.shapes, after.shapes):
        assert b.provenance["quadrature_subdivisions"] == 5
        assert len(b.samples) == len(a.faces) * 3 * 4**5
        for name in ("vertices", "faces", "sdf", "origin", "inertia"):
            assert np.array_equal(getattr(a, name), getattr(b, name))
