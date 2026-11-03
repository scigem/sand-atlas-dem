"""Analytical box contacts independently check the source-pose study algorithm."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh

from sand_atlas_dem.geometry import from_mesh, with_face_quadrature

source = Path(__file__).resolve().parents[1] / "examples/library_contact_poses.py"
spec = importlib.util.spec_from_file_location("library_contact_poses", source)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.mark.parametrize("extents", [[1, 1, 1], [1, 2, 3]])
def test_frozen_pose_bisection_and_traction_match_analytical_boxes(extents):
    shape = with_face_quadrature(
        from_mesh(
            trimesh.creation.box(extents=extents),
            unit="mm",
            resolution=8,
            samples=16,
            provenance={"source_sha256": "original-analytical-box-fixture"},
        )
    )
    policy = dict(
        target_sampled_depth_fractions=[0.0025, 0.01],
        bracket_scan_intervals=256,
        bisection_iterations=32,
        maximum_depth_shortfall_fraction=1e-8,
        maximum_bracket_width_fraction=1e-8,
        query_batch_size=16,
    )
    unit = (6 * shape.volume / np.pi) ** (1 / 3)
    for row in study.prepare_shape(shape, policy):
        depth = row["target_sampled_depth_fraction"] * unit
        assert row["separation_m"] == pytest.approx(
            extents[0] * 0.001 - depth, abs=unit * 1e-8
        )
        expected = [-extents[1] * extents[2] * 1e-6 * depth, 0, 0]
        # Compare the observable vector in its own force scale. Componentwise
        # relative checks at an exact zero otherwise impose an unrelated
        # dimensionful floor on FP64 transverse cancellation.
        actual = np.asarray(row["exact_normal_traction_integral_m3"])
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 1e-7
        assert row["exact_active_nodes"] == [6, 6]
