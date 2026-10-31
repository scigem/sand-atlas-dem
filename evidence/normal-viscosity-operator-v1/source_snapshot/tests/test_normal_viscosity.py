"""Prospective traction-viscosity checks, frozen before first Metal execution."""

import os

import numpy as np
import pytest

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, sphere
from sand_atlas_dem.gpu import DAMPING_STATE, Engine, RejectedStep, f4
from sand_atlas_dem.spec import Experiment

metal_test = pytest.mark.skipif(
    os.environ.get("SAND_DEM_METAL") != "1", reason="Serial Metal required"
)


@pytest.fixture(scope="module")
def metal():
    return Metal()


def test_viscosity_spec_and_checkpoint_identity():
    assert Experiment().checkpoint_format == "sand-dem-checkpoint/3"
    for value in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            Experiment(normal_viscosity=value)
    value = Experiment(normal_viscosity=1e6)
    assert value.viscous and value.checkpoint_format == "sand-dem-checkpoint/5"
    assert Experiment(**value.to_dict()).normal_viscosity == 1e6


@pytest.mark.metal
@metal_test
@pytest.mark.parametrize("weight", [0.5, 1.0])
@pytest.mark.parametrize("viscosity", [0.0, 3.0])
def test_normal_operator_compression_unload_clamp_inactive(metal, weight, viscosity):
    # Hand-evaluated table: compression, elastic, unclipped unloading, clipped
    # unloading, separation and exact contact. Stiffness=8, area=2, depth=0.25.
    data = np.array(
        [[0.25, 2, v, 0] for v in (-2, 0, 0.25, 2)] + [[-0.1, 2, -2, 0], [0, 2, -2, 0]],
        np.float32,
    )
    expected = (
        np.array(
            [[4, 0], [4, 0], [4, 0], [4, 0], [0, 0], [0, 0]]
            if viscosity == 0
            else [[16, 24], [4, 0], [2.5, 0.375], [0, 8], [0, 0], [0, 0]]
        )
        * weight
    )
    output = metal.buffer(count=len(data), stride=16)
    metal.kernel("normalProbe", "normal.slang").dispatch(
        thread_count=[len(data), 1, 1],
        N=len(data),
        stiffness=8,
        viscosity=viscosity,
        weight=weight,
        input=metal.buffer(data),
        output=output,
    )
    metal.device.wait()
    actual = metal.read(output, shape=(len(data), 4))[:, :2]
    assert np.allclose(actual, expected, rtol=2e-6, atol=1e-7)
    assert (actual >= 0).all()


def cap_engine(metal, *, viscosity=1e6, velocity=-0.1, samples=32768):
    shape = sphere(resolution=16, samples=samples)
    wall = dict(
        normal=[0, 0, 1],
        offset=0.00155,
        velocity=[0, 0, 0],
        normal_stiffness=1e12,
        tangential_stiffness=5e11,
        friction=0,
        normal_viscosity=viscosity,
    )
    spec = Experiment(
        periodic=False,
        particles=1,
        timestep=1e-8,
        max_overlap_fraction=0.1,
        contacts_per_particle=4096,
        planes=[wall],
    )
    engine = Engine(
        ShapeLibrary([shape]), spec, [[0.002] * 3], [0.006] * 3, [0], [1], metal=metal
    )
    engine.buffers["velocity"].copy_from_numpy(
        f4([[0, 0, velocity * engine.time_unit / engine.length_unit]])
    )
    engine.initialize()
    return engine


@pytest.mark.metal
@metal_test
def test_cap_continuum_force_power_and_common_moment(metal):
    engine = cap_engine(metal)
    radius, depth, speed, stiffness, viscosity = 0.0005, 0.00005, 0.1, 1e12, 1e6
    expected = (
        np.pi * stiffness * radius * depth**2
        + 2 * np.pi * viscosity * radius * depth * speed
    )
    power = 2 * np.pi * viscosity * radius * depth * speed**2
    state, row = engine.state(), engine.diagnostics()
    force = state["force"][0, :3] * engine.force_unit
    assert abs(force[2] / expected - 1) < 0.001
    assert abs(row["normal_contact_power"] / power - 1) < 0.001
    assert row["normal_contact_dissipation"] == 0
    reaction = np.array(row["walls"][0]["reaction"])
    assert np.linalg.norm(force + reaction) < expected * 1e-5
    moment = np.cross([0.002] * 3, force) + state["torque"][0, :3] * engine.energy_unit
    assert (
        np.linalg.norm(moment + row["walls"][0]["reaction_moment_origin"])
        < expected * 0.006 * 1e-5
    )


@pytest.mark.metal
@metal_test
def test_viscous_restart_preserves_force_timing_and_loss(metal):
    engine = cap_engine(metal)
    engine.advance(5)
    saved = engine.state()
    assert engine.diagnostics()["normal_contact_dissipation"] > 0
    assert not np.array_equal(saved["forceVelocity"], saved["velocity"])
    engine.advance(7)
    expected, expected_loss = (
        engine.state(),
        engine.diagnostics()["normal_contact_dissipation"],
    )
    engine.restore(saved, 5)
    engine.advance(7)
    actual = engine.state()
    for name in (
        "position",
        "velocity",
        "rotation",
        "momentum",
        "force",
        "torque",
        *DAMPING_STATE,
    ):
        assert np.allclose(actual[name], expected[name], rtol=2e-4, atol=3e-7), name
    assert engine.diagnostics()["normal_contact_dissipation"] == pytest.approx(
        expected_loss, rel=2e-4
    )
    before = engine.state()
    with pytest.raises(RejectedStep, match="penetration"):
        engine.advance(3, wall_velocities=[[0, 0, 1e5]])
    for name, value in before.items():
        assert np.array_equal(engine.state()[name], value), name
    assert engine.step == 12


@pytest.mark.metal
@metal_test
def test_viscous_timestep_preflight(metal):
    with pytest.raises(ValueError, match="Timestep exceeds conservative"):
        cap_engine(metal, viscosity=1e12, samples=64)
