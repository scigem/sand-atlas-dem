"""Independent cap integrals verify production Coulomb force, torque and loss."""

import os

import numpy as np
import pytest

from sand_atlas_dem.geometry import ShapeLibrary, sphere
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(
        os.environ.get("SAND_DEM_METAL") != "1", reason="Serial Metal required"
    ),
]


def test_production_wall_partial_slip_matches_cap_force_torque_and_return_loss():
    radius, kn, kt, mu, slip = 0.0005, 1e12, 5e11, 0.3, 0.000025
    dt = 1e-8
    shape = sphere(resolution=16, samples=32768)
    wall = dict(
        normal=[0, 0, 1],
        offset=0.00155,
        velocity=[0, 0, 0],
        normal_stiffness=kn,
        tangential_stiffness=kt,
        friction=mu,
    )
    spec = Experiment(
        periodic=False,
        particles=1,
        timestep=dt,
        planes=[wall],
        max_overlap_fraction=0.1,
        contacts_per_particle=4096,
    )
    engine = Engine(ShapeLibrary([shape]), spec, [[0.002] * 3], [0.006] * 3, [0], [1])
    engine.initialize()
    engine.advance(1, wall_velocities=[[slip / dt, 0, 0]])
    state, row = engine.state(), engine.diagnostics()
    centre = (
        state["position"][0, :3].astype(float) + state["positionRemainder"][0, :3]
    ) * engine.length_unit
    depth = radius + wall["offset"] - centre[2]
    # dA=2*pi*R*dd. Trial spring=-slip*x and traction=min(kt*slip,mu*kn*d).
    c, a = mu * kn, kt * slip
    h = min(depth, a / c)
    j0 = c * h**2 / 2 + a * (depth - h)
    j1 = c * h**3 / 3 + a * (depth**2 - h**2) / 2
    expected_force = np.array(
        [2 * np.pi * radius * j0, 0, np.pi * kn * radius * depth**2]
    )
    expected_torque = np.array(
        [0, -2 * np.pi * radius * ((radius - depth) * j0 + j1), 0]
    )
    expected_loss = 2 * np.pi * radius * c * (slip * h**2 / 2 - c * h**3 / (3 * kt))
    force = state["force"][0, :3] * engine.force_unit
    torque = state["torque"][0, :3] * engine.energy_unit
    assert (
        np.linalg.norm(force - expected_force) < np.linalg.norm(expected_force) * 0.001
    )
    assert (
        np.linalg.norm(torque - expected_torque)
        < np.linalg.norm(expected_torque) * 0.001
    )
    assert abs(row["friction_dissipation"] / expected_loss - 1) < 0.001
    assert 0 < row["sliding_fraction"] < 1
    assert row["wall_work"] > 0
