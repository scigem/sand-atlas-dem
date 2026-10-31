"""Prospective nonperiodic neighbour/motion/transaction acceptance cases."""

import os

import numpy as np
import pytest

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, sphere
from sand_atlas_dem.gpu import Engine, RejectedStep, f4
from sand_atlas_dem.spec import Experiment

metal_test = [
    pytest.mark.metal,
    pytest.mark.skipif(
        os.environ.get("SAND_DEM_METAL") != "1", reason="Serial Metal required"
    ),
]


def test_nonperiodic_contract():
    assert Experiment().periodic is True
    assert Experiment(periodic=False).to_dict()["periodic"] is False
    for value in (0, 1, "false", None):
        with pytest.raises(ValueError, match="boolean"):
            Experiment(periodic=value)
    for kind in ("strain", "isotropic", "triaxial"):
        with pytest.raises(ValueError, match="Nonperiodic"):
            Experiment(periodic=False, stages=[dict(kind=kind, steps=1)])
    for point in ([-1e-9, 0, 0], [0.006, 0, 0], [0, float("nan"), 0]):
        with pytest.raises(ValueError):
            Engine(
                None,
                Experiment(periodic=False),
                [point],
                [0.006] * 3,
                [0],
                [1],
                metal=object(),
            )


@pytest.fixture(scope="module")
def metal():
    return Metal()


@pytest.fixture(scope="module")
def library():
    return ShapeLibrary([sphere(resolution=32, samples=512)])


def make_engine(metal, library, positions, cell=(0.006, 0.006, 0.006), **settings):
    spec = Experiment(
        particles=len(positions), periodic=False, timestep=1e-8, **settings
    )
    return Engine(
        library,
        spec,
        positions,
        cell,
        np.zeros(len(positions), np.uint32),
        np.ones(len(positions)),
        metal=metal,
    )


def device_pairs(engine):
    count = int(engine.check_errors()[1])
    return engine.m.read(engine.buffers["pairs"], np.uint32).reshape(-1, 2)[:count]


@pytest.mark.parametrize("small_box", [False, True])
@pytest.mark.metal
@pytest.mark.skipif(
    os.environ.get("SAND_DEM_METAL") != "1", reason="Serial Metal required"
)
def test_bounded_bins_match_independent_all_pairs(metal, library, small_box):
    # Lattice includes boundary bins and a one-bin domain. No grain intersections.
    axes = [0.0001, 0.00113, 0.00216] if not small_box else [0.0001, 0.00113]
    positions = np.array(np.meshgrid(axes, axes, axes)).reshape(3, -1).T
    cell = np.array([0.0047] * 3 if not small_box else [0.0015] * 3)
    engine = make_engine(metal, library, positions, cell)
    engine.initialize()
    actual = device_pairs(engine)
    radius = library.shapes[0].radius
    cutoff = 2 * radius * (1 + engine.spec.skin)
    expected = {
        (i, j)
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
        if np.linalg.norm(positions[j] - positions[i]) <= cutoff
    }
    assert len(actual) == len(expected)  # Also detects duplicate bin visits.
    assert {tuple(pair) for pair in actual} == expected
    assert engine.diagnostics()["active_surface_contacts"] == 0


@pytest.mark.metal
@pytest.mark.skipif(
    os.environ.get("SAND_DEM_METAL") != "1", reason="Serial Metal required"
)
def test_opposite_faces_are_not_neighbours(metal, library):
    positions = [[0.0001, 0.002, 0.002], [0.00511, 0.002, 0.002]]
    engine = make_engine(metal, library, positions)
    engine.initialize()
    assert len(device_pairs(engine)) == 0
    assert engine.diagnostics()["active_surface_contacts"] == 0
    assert np.array_equal(engine.state()["force"], np.zeros((2, 4)))


@pytest.mark.parametrize(
    "axis,sign", [(axis, sign) for axis in range(3) for sign in (-1, 1)]
)
@pytest.mark.metal
@pytest.mark.skipif(
    os.environ.get("SAND_DEM_METAL") != "1", reason="Serial Metal required"
)
def test_domain_escape_rolls_back_entire_chunk(metal, library, axis, sign):
    positions = np.array([[0.003] * 3])
    positions[0, axis] = 1e-7 if sign < 0 else 0.006 - 1e-7
    engine = make_engine(metal, library, positions)
    engine.initialize()
    v = np.zeros((1, 3))
    v[0, axis] = sign * 4  # 40 nm/step; escape during the five-step transaction.
    engine.buffers["velocity"].copy_from_numpy(
        f4(v * engine.time_unit / engine.length_unit)
    )
    before = engine.state()
    with pytest.raises(RejectedStep, match="domain escape") as error:
        engine.advance(5)
    assert error.value.flags == 2048
    after = engine.state()
    for name in before:
        assert np.array_equal(before[name], after[name]), name
    assert engine.step == 0
    engine.buffers["velocity"].copy_from_numpy(np.zeros((1, 4), np.float32))
    engine.advance(2)  # Engine remains usable after rejection.
    assert engine.step == 2


@pytest.mark.metal
@pytest.mark.skipif(
    os.environ.get("SAND_DEM_METAL") != "1", reason="Serial Metal required"
)
def test_nonperiodic_drift_rebuild_and_restore(metal, library):
    engine = make_engine(metal, library, [[0.002, 0.002, 0.002]])
    engine.initialize()
    velocity = np.array([[100.0, -20.0, 10.0]])
    engine.buffers["velocity"].copy_from_numpy(
        f4(velocity * engine.time_unit / engine.length_unit)
    )
    initial = engine.state()["position"][0, :3] * engine.length_unit
    engine.advance(40)
    saved = engine.state()
    assert (
        np.linalg.norm(
            saved["position"][0, :3]
            - engine.m.read(engine.buffers["referencePosition"]).reshape(-1, 4)[0, :3]
        )
        < engine.skin / 2
    )
    assert np.allclose(
        (saved["position"][0, :3] + saved["positionRemainder"][0, :3])
        * engine.length_unit,
        initial + velocity[0] * 40e-8,
        rtol=0,
        atol=3e-10,
    )
    engine.advance(10)
    expected = engine.state()
    engine.restore(saved, 40)
    engine.advance(10)
    for name, value in expected.items():
        assert np.array_equal(engine.state()[name], value), name
    before = engine.state()
    with pytest.raises(ValueError, match="periodic"):
        engine.advance(1, rate=[1, 0, 0])
    for name, value in before.items():
        assert np.array_equal(engine.state()[name], value), name
