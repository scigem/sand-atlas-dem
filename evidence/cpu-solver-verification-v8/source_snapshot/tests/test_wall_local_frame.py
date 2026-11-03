"""Physical translation invariance of shallow normal contact, including oblique planes."""

import os

import numpy as np
import pytest
import trimesh

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(
        os.environ.get("SAND_DEM_METAL") != "1",
        reason="Explicit serial Metal execution required",
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


@pytest.mark.parametrize("centre", [0.005, 5.0])
@pytest.mark.parametrize("normal", [[0, 0, 1], [0.6, 0, 0.8], [1 / 3, 2 / 3, 2 / 3]])
def test_shallow_contact_preserves_force_and_local_torque_under_translation(
    metal, cube, centre, normal
):
    h, depth, kn = 1e-4, 1e-7, 1e12
    p = np.full(3, centre)
    n = np.array(normal)
    offset = float(p @ n + min(cube.samples @ n) + depth)
    wall = dict(
        normal=normal,
        offset=offset,
        velocity=[0, 0, 0],
        normal_stiffness=kn,
        tangential_stiffness=kn / 2,
        friction=0,
    )
    spec = Experiment(
        periodic=False,
        particles=1,
        timestep=2e-9,
        planes=[wall],
        contacts_per_particle=64,
        pairs_per_particle=8,
        bin_capacity=16,
        max_overlap_fraction=0.1,
    )
    e = Engine(
        ShapeLibrary([cube]),
        spec,
        [p],
        [max(0.01, 2 * centre)] * 3,
        [0],
        [1],
        metal=metal,
    )
    e.initialize()
    state = e.state()
    delta = np.maximum(0, offset - p @ n - cube.samples @ n)
    forces = kn * (cube.areas * delta)[:, None] * n
    expected = forces.sum(axis=0)
    torque = np.cross(cube.samples, forces).sum(axis=0)
    scale = np.linalg.norm(expected)
    actual = state["force"][0, :3].astype(float) * e.force_unit
    actual_torque = state["torque"][0, :3].astype(float) * e.force_unit * e.length_unit
    assert np.linalg.norm(actual - expected) / scale <= 1e-4
    assert np.linalg.norm(actual_torque - torque) / (scale * h) <= 1e-4
    restored = (
        state["position"][0, :3].astype(float) + state["positionRemainder"][0, :3]
    ) * e.length_unit
    assert np.linalg.norm(restored - p) / h <= 1e-7
    assert len(state["meta"]) == np.count_nonzero(delta)
