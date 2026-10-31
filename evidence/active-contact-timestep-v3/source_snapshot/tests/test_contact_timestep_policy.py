"""Prospective active-contact gates: algebra, device rates, rollback and refinement.

Frozen tolerances: rates 2e-5 relative, trajectory position/d <= 1e-3,
velocity/sqrt(kn*d/rho) <= 1e-3, quaternion geodesic <= 1e-3 rad.
All GPU trajectories use Metal; no CPU trajectory substitutes.
"""

import os

import numpy as np
import pytest
import trimesh

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh, with_face_quadrature
from sand_atlas_dem.gpu import Engine, RejectedStep, f4
from sand_atlas_dem.spec import Experiment


def test_policy_explicit_and_roundtrips():
    assert Experiment().timestep_policy == "conservative"
    spec = Experiment(timestep_policy="active-contact")
    assert Experiment(**spec.to_dict()).timestep_policy == "active-contact"
    with pytest.raises(ValueError, match="timestep policy"):
        Experiment(timestep_policy="automatic")


def test_pair_coupling_spectral_bound():
    # Independent assembled mass-normalized spring matrix, with rotational DOFs.
    rng = np.random.default_rng(7301)
    n = 5
    matrix = np.zeros((6 * n, 6 * n))
    sums = np.zeros(n)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]:
        a, b = rng.normal(size=(2, 6))
        k = rng.uniform(0.1, 2)
        row = np.zeros(6 * n)
        row[6 * i : 6 * i + 6], row[6 * j : 6 * j + 6] = a, -b
        matrix += k * np.outer(row, row)
        sums[i] += 2 * k * np.dot(a, a)
        sums[j] += 2 * k * np.dot(b, b)
    assert np.linalg.eigvalsh(matrix).max() <= sums.max()
    # Using only one endpoint or omitting coupling is insufficient in general.
    assert np.linalg.eigvalsh(np.array([[1.0, -1.0], [-1.0, 1.0]])).max() == 2


metal_test = [
    pytest.mark.metal,
    pytest.mark.skipif(
        os.environ.get("SAND_DEM_METAL") != "1", reason="Explicit serial Metal required"
    ),
]


@pytest.fixture(scope="module")
def metal():
    return Metal()


@pytest.fixture(scope="module")
def cube():
    h = 1e-4
    shape = from_mesh(
        trimesh.creation.box(extents=[h] * 3), unit="m", resolution=8, samples=16
    )
    shape.samples = np.concatenate([np.eye(3), -np.eye(3)]) * h / 2
    shape.areas = np.full(6, h * h)
    shape.validate()
    return shape


def engine(metal, cube, dt=1e-8, viscosity=0, walls=2, positions=None):
    centre, h, depth = 0.005, 1e-4, 1e-7
    planes = [
        dict(
            normal=[0, 0, 1],
            offset=centre - h / 2 + depth,
            velocity=[0, 0, 0],
            normal_stiffness=1e12,
            tangential_stiffness=5e11,
            friction=0.5,
            normal_viscosity=viscosity,
        )
    ]
    if walls == 2:
        planes.append(
            dict(planes[0], normal=[0, 0, -1], offset=-(centre + h / 2 - depth))
        )
    if walls == 0:
        planes = []
    positions = positions if positions is not None else [[centre] * 3]
    return Engine(
        ShapeLibrary([cube]),
        Experiment(
            periodic=False,
            contact_geometry="mesh",
            particles=len(positions),
            timestep=dt,
            timestep_policy="active-contact",
            planes=planes,
            normal_viscosity=viscosity,
            max_overlap_fraction=0.1,
            contacts_per_particle=128,
            pairs_per_particle=16,
            bin_capacity=32,
        ),
        positions,
        [0.01] * 3,
        [0] * len(positions),
        [1] * len(positions),
        metal=metal,
    )


@pytest.mark.parametrize("viscosity", [0, 1e6])
@pytest.mark.parametrize("walls", [1, 2])
@pytest.mark.metal
@metal_test[1]
def test_flat_wall_rates_and_restart(metal, cube, viscosity, walls):
    e = engine(metal, cube, viscosity=viscosity, walls=walls, dt=1e-10)
    e.initialize()
    # h^2 area, h/2 lever arm, isotropic cube I=m*h^2/6.
    mobility = (1 + 1.5) / (2650 * 1e-12)
    expected = walls * 1e-8 * 2e12 * mobility
    rates = e.m.read(e.buffers["timestepRows"], shape=(1, 4))[0]
    assert abs(rates[0] / (expected * e.time_unit**2) - 1) <= 2e-5
    if viscosity:
        expected_c = walls * 1e-8 * viscosity * 1.5 * mobility
        assert abs(rates[1] / (expected_c * e.time_unit) - 1) <= 2e-5
    state = e.state()
    e.advance(4)
    e.restore(state, 0)
    np.testing.assert_allclose(
        e.timestep_limits()["elastic_seconds"], [0.1 / np.sqrt(expected)], rtol=2e-5
    )


@pytest.mark.metal
@metal_test[1]
def test_oversized_step_and_spin_reject_atomically(metal, cube):
    e = engine(metal, cube, dt=1e-6)
    e.initialize()
    before = e.state()
    with pytest.raises(RejectedStep) as exc:
        e.advance(3)
    assert exc.value.flags & 8192
    after = e.state()
    assert e.step == 0
    for key in before:
        np.testing.assert_array_equal(after[key], before[key], err_msg=key)
    e.spec.timestep = 1e-8
    e.buffers["momentum"].copy_from_numpy(f4([[0, 0, 1e6]]))
    with pytest.raises(RejectedStep) as exc:
        e.advance(1)
    assert exc.value.flags & 8192


@pytest.mark.metal
@metal_test[1]
def test_new_contact_rejects_and_rolls_back(metal, cube):
    # Initially separated, then plane creates contact during the drift. Small
    # allowed wall travel, but the newly activated spring rejects the large dt.
    e = engine(
        metal, cube, walls=1, dt=1e-7, positions=[[0.005, 0.005, 0.005 + 1.1e-7]]
    )
    e.initialize()
    assert e.state()["contacts"] == 0
    before = e.state()
    with pytest.raises(RejectedStep) as exc:
        e.advance(1, wall_velocities=np.array([[0, 0, 0.2]]))
    assert exc.value.flags & 8192
    for key, value in before.items():
        np.testing.assert_array_equal(e.state()[key], value, err_msg=key)
    assert e.step == 0


@pytest.mark.metal
@metal_test[1]
@pytest.mark.parametrize("dense", [False, True])
def test_coupled_rotating_chain_refinement(metal, cube, dense):
    # Three grains, two contacts/body in the middle, sticking/sliding plus spin.
    positions = [[0.005 + i * 0.999e-4, 0.005, 0.005] for i in range(3)]
    if dense:
        positions = 0.005 + np.indices((3, 3, 3)).reshape(3, -1).T * 0.999e-4
    n = len(positions)
    outputs = []
    for dt, steps in [(1e-9, 2048), (2e-9, 1024), (4e-9, 512), (8e-9, 256)]:
        e = engine(metal, cube, dt=dt, walls=0, positions=positions)
        e.buffers["velocity"].copy_from_numpy(
            f4(np.array([[0, 0.001 * np.cos(i), 0.001 * np.sin(i)] for i in range(n)]))
            * e.time_unit
            / e.length_unit
        )
        # Nonzero angular momentum: physical 1000 rad/s about an oblique axis.
        inertia = cube.inertia * 2650
        L = (
            inertia
            @ np.array([600.0, 800.0, 0.0])
            / (e.mass_unit * e.length_unit**2 / e.time_unit)
        )
        e.buffers["momentum"].copy_from_numpy(f4([L * (-1) ** i for i in range(n)]))
        e.initialize()
        assert e.state()["contacts"] >= (108 if dense else 4)
        if dense and dt == 8e-9:
            before = e.state()
            with pytest.raises(RejectedStep) as exc:
                e.advance(steps)
            assert exc.value.flags & 8192
            for key, value in before.items():
                np.testing.assert_array_equal(e.state()[key], value, err_msg=key)
            continue
        e.advance(steps)
        outputs.append((e.state(), e.length_unit, e.time_unit))
    ref, l, t = outputs[0]
    for state, _, _ in outputs[1:]:
        assert (
            np.max(
                np.linalg.norm(
                    (state["position"][:, :3] + state["positionRemainder"][:, :3])
                    - (ref["position"][:, :3] + ref["positionRemainder"][:, :3]),
                    axis=1,
                )
            )
            * l
            / 1e-4
            <= 1e-3
        )
        assert (
            np.max(
                np.linalg.norm(
                    state["velocity"][:, :3] - ref["velocity"][:, :3], axis=1
                )
            )
            * l
            / t
            / np.sqrt(1e12 * 1e-4 / 2650)
            <= 1e-3
        )
        q, qr = state["rotation"].astype(float), ref["rotation"].astype(float)
        q /= np.linalg.norm(q, axis=1)[:, None]
        qr /= np.linalg.norm(qr, axis=1)[:, None]
        assert np.max(2 * np.arccos(np.clip(abs(np.sum(q * qr, axis=1)), 0, 1))) <= 1e-3


@pytest.mark.metal
@metal_test[1]
def test_active_patch_allows_larger_step(metal, record_property):
    sh = with_face_quadrature(
        from_mesh(
            trimesh.creation.icosphere(subdivisions=2, radius=5e-5),
            unit="m",
            resolution=8,
            samples=32,
        ),
        rule="gauss3",
        subdivisions=2,
    )
    e = engine(metal, sh, dt=1e-8, walls=1)
    e.initialize()
    limit = float(e.timestep_limits()["elastic_seconds"][0])
    # Independently reconstruct legacy whole-surface preflight, same shape/material.
    stiffness = 3e12 * sh.areas.sum()
    old = 0.1 * np.sqrt(
        min(
            2650 * sh.volume / stiffness,
            np.linalg.eigvalsh(sh.inertia * 2650).min() / (stiffness * sh.radius**2),
        )
    )
    assert limit > 2 * old
    assert e.state()["contacts"] > 0
    e.spec.timestep = min(2 * old, limit / 2)
    e.advance(32)
    record_property("initial_contact_limit_seconds", limit)
    record_property("legacy_limit_seconds", old)
    record_property("initial_limit_ratio", limit / old)
