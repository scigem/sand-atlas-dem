import numpy as np
import pytest

from sand_atlas_dem.annealing import agitation, cooling_energy, select_timestep


def test_anisotropic_agitation_energy_and_momentum():
    mass = np.array([1., 2., 7.])
    inertia = np.array([[[3., .2, 0], [.2, 2., 0], [0, 0, 1.]]] * 3)
    v, angular = agitation(mass, inertia, .04, np.random.default_rng(3))
    np.testing.assert_allclose((mass[:, None] * v).sum(0), 0, atol=1e-15)
    translation = .5 * (mass[:, None] * v**2).sum()
    rotation = .5 * np.einsum('ni,ni->', angular, np.linalg.solve(inertia, angular[..., None])[..., 0])
    assert translation == pytest.approx(3 * .04 / 2)
    assert rotation == pytest.approx(translation)


def test_schedule_and_zero_agitation():
    assert cooling_energy(0, 50, 100, .04) == .04
    assert cooling_energy(100, 50, 100, .04) == pytest.approx(.01)
    assert cooling_energy(150, 50, 100, .04) == 0
    v, angular = agitation([1, 2], [np.eye(3)] * 2, 0, np.random.default_rng(1))
    assert not np.any(v) and not np.any(angular)


def test_timestep_uses_most_restrictive_body_and_viscosity():
    assert select_timestep([2e-6, 4e-6], [np.inf, 1e-6], 3e-6) == 5e-7
    assert select_timestep([np.inf], [np.inf], 1e-6) == 1e-6
    with pytest.raises(ValueError):
        select_timestep([float('nan')], [np.inf], 1e-6)


def test_invalid_inertia_and_energy_rejected():
    with pytest.raises(ValueError):
        agitation([1, 2], [-np.eye(3)] * 2, .1, np.random.default_rng(1))
    with pytest.raises(ValueError):
        cooling_energy(0, 1, 2, -1)


def test_construction_quadrature_preserves_mesh_area_and_reproducibility():
    from sand_atlas_dem.annealing import construction_quadrature
    from sand_atlas_dem.geometry import sphere

    shape = sphere(samples=128, resolution=8)
    coarse, indices = construction_quadrature(shape, 32, 7)
    again, again_indices = construction_quadrature(shape, 32, 7)
    assert coarse.areas.sum() == pytest.approx(shape.areas.sum(), rel=1e-14)
    np.testing.assert_array_equal(coarse.samples, shape.samples[indices])
    np.testing.assert_array_equal(indices, again_indices)
    np.testing.assert_array_equal(coarse.samples, again.samples)
    for name in ['vertices', 'faces', 'inertia', 'sdf']:
        np.testing.assert_array_equal(getattr(coarse, name), getattr(shape, name))
    assert 'construction_quadrature' not in shape.provenance
