"""Balanced source-shape inventory without changing legacy random initialization."""

from dataclasses import replace

import numpy as np
import pytest

from sand_atlas_dem.geometry import ShapeLibrary, sphere
from sand_atlas_dem.simulation import initial_assembly
from sand_atlas_dem.spec import Experiment


@pytest.fixture
def library():
    source = sphere(resolution=8, samples=16)
    return ShapeLibrary([replace(source, id=f"shape-{i}") for i in range(3)])


@pytest.mark.parametrize("particles", [3, 7, 24])
def test_balanced_inventory_seed_and_nonoverlap(library, particles):
    spec = Experiment(particles=particles, shape_sampling="balanced", cell=[0.02] * 3)
    a = initial_assembly(library, spec)
    b = initial_assembly(library, spec)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    positions, cell, ids, scales = a
    counts = np.bincount(ids, minlength=3)
    assert counts.min() >= 1 and counts.max() - counts.min() <= 1
    assert ids.dtype == np.uint32 and np.all(scales == 1)
    assert np.all(positions >= 0) and np.all(positions < cell)
    for i in range(particles):
        branches = positions[i + 1 :] - positions[i]
        branches -= np.round(branches / cell) * cell
        assert np.all(np.linalg.norm(branches, axis=1) > 2 * library.shapes[0].radius)
    changed = initial_assembly(library, replace(spec, seed=spec.seed + 1))
    assert not np.array_equal(a[0], changed[0])


def test_balanced_rejects_missing_shapes_and_unknown_mode(library):
    with pytest.raises(ValueError, match="at least one particle per shape"):
        initial_assembly(library, Experiment(particles=2, shape_sampling="balanced"))
    with pytest.raises(ValueError, match="shape_sampling"):
        Experiment(shape_sampling="guessed-population")


def test_default_random_shape_sequence_is_unchanged(library):
    spec = Experiment(particles=24, seed=17, cell=[0.02] * 3)
    ids = initial_assembly(library, spec)[2]
    expected = np.random.default_rng(17).integers(3, size=24, dtype=np.uint32)
    assert np.array_equal(ids, expected)
    assert Experiment(**spec.to_dict()).shape_sampling == "random"
