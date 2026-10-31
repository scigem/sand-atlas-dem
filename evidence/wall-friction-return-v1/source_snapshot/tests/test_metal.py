import os

import numpy as np
import pytest

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, sphere
from sand_atlas_dem.gpu import Engine, Scan
from sand_atlas_dem.reference import pair_contact
from sand_atlas_dem.spec import Experiment

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(
        os.environ.get("SAND_DEM_METAL") != "1",
        reason="Set SAND_DEM_METAL=1 to explicitly execute compiled Metal tests serially",
    ),
]


@pytest.fixture(scope="module")
def metal():
    return Metal()


@pytest.fixture(scope="module")
def shape():
    return sphere(resolution=32, samples=512)


def engine(metal, shape, **kwargs):
    spec = Experiment(
        particles=2,
        timestep=1e-8,
        max_overlap_fraction=0.03,
        contacts_per_particle=128,
        **kwargs,
    )
    x = np.array([[0.002, 0.002, 0.002], [0.00299, 0.002, 0.002]])
    return Engine(
        ShapeLibrary([shape]), spec, x, [0.006] * 3, [0, 0], [1.0, 1.0], metal=metal
    ), x


def test_real_device(metal):
    assert metal.doctor()["backend"] == "metal"
    assert "Apple" in metal.doctor()["adapter"]


def test_hierarchical_scan(metal):
    rng = np.random.default_rng(8)
    for n in [1, 255, 256, 257, 10000]:
        data = rng.integers(0, 20, n, dtype=np.uint32)
        a = metal.buffer(data)
        b = metal.buffer(count=n)
        enc = metal.device.create_command_encoder()
        Scan(metal, n).encode(enc, a, b)
        metal.submit(enc)
        metal.device.wait()
        assert np.array_equal(
            metal.read(b, np.uint32), np.cumsum(data, dtype=np.uint32) - data
        )


def test_contact_reference_and_balance(metal, shape):
    e, x = engine(metal, shape)
    e.initialize()
    force = e.m.read(e.buffers["force"]).reshape(-1, 4)[:, :3] * e.force_unit
    torque = e.m.read(e.buffers["torque"]).reshape(-1, 4)[:, :3] * e.energy_unit
    r = pair_contact(
        shape, shape, *x, kn=e.spec.normal_stiffness, kt=e.spec.tangential_stiffness
    )
    assert np.linalg.norm(force[0] - r["force"]) / np.linalg.norm(r["force"]) < 1e-4
    assert np.allclose(force[0], -force[1], rtol=1e-5, atol=1e-9)
    assert np.allclose(torque[0], r["torque_i"], rtol=1e-4, atol=1e-9)
    assert np.linalg.norm(torque.sum(0) + np.cross(x[0] - x[1], force[0])) < 1e-8
    diagnostic = e.diagnostics()
    assert diagnostic["active_surface_contacts"] == r["contacts"] > 0
    assert diagnostic["active_grain_pairs"] == 1
    assert diagnostic["coordination"] == 1
    assert np.isclose(np.trace(diagnostic["fabric"]), 1)
    assert diagnostic["friction_dissipation"] == 0


def test_periodic_pair_and_history_restart(metal, shape):
    e, x = engine(metal, shape)
    x = np.array([[0.0001, 0.002, 0.002], [0.00511, 0.002, 0.002]])
    e.buffers["position"].copy_from_numpy(
        np.column_stack([x / e.length_unit, np.zeros(2)]).astype(np.float32)
    )
    e.initialize()
    assert e.diagnostics()["active_surface_contacts"] > 0
    velocity = np.zeros((2, 4), np.float32)
    velocity[0, 1] = 0.01
    e.buffers["velocity"].copy_from_numpy(velocity)
    e.advance(3)
    state = e.state()
    e.advance(2)
    expected = e.state()
    e.restore(state, 3)
    e.advance(2)
    actual = e.state()
    for name in ["position", "velocity", "rotation", "momentum", "force", "torque"]:
        assert np.allclose(actual[name], expected[name], rtol=2e-4, atol=3e-7), name


def test_overlap_rollback(metal, shape):
    e, x = engine(metal, shape)
    e.initialize()
    before = e.state()
    with pytest.raises(RuntimeError, match="penetration"):
        e.advance(1, rate=[-1e8, 0, 0])
    after = e.state()
    for name in [
        "position",
        "velocity",
        "rotation",
        "momentum",
        "meta",
        "slip",
        "normalArea",
        "cell",
        "positionRemainder",
        "velocityRemainder",
        "momentumRemainder",
        "rotationRemainder",
        "accumulatedRemainder",
        "cellLog",
    ]:
        assert np.array_equal(before[name], after[name]), name
    assert e.step == 0


def test_affine_cell_and_no_particle_fallback(metal, shape):
    spec = Experiment(particles=2, timestep=1e-8)
    x = np.array([[0.001, 0.001, 0.001], [0.004, 0.004, 0.004]])
    e = Engine(ShapeLibrary([shape]), spec, x, [0.006] * 3, [0, 0], [1, 1], metal=metal)
    e.initialize()
    e.advance(5, rate=[-1000, 0, 0])
    state = e.state()
    assert np.allclose(
        state["cell"][0, 0] * e.length_unit, 0.006 * np.exp(-1000 * 5e-8), rtol=1e-6
    )
    assert e.diagnostics()["candidate_pairs"] == 0


def test_tangential_force_and_concave_mesh_reference(metal):
    from pathlib import Path

    p = Path("assets/generated/concave-prism.h5")
    if not p.exists():
        pytest.skip("Generate original fixtures with examples/create_fixtures.py")
    shape = ShapeLibrary.load(p).shapes[0]
    spec = Experiment(
        particles=2, timestep=1e-9, max_overlap_fraction=0.3, contacts_per_particle=512
    )
    x = np.array([[0.002, 0.002, 0.002], [0.0028, 0.002, 0.002]])
    e = Engine(ShapeLibrary([shape]), spec, x, [0.008] * 3, [0, 0], [1, 1], metal=metal)
    e.initialize()
    assert e.diagnostics()["active_surface_contacts"] > 0
    force = e.m.read(e.buffers["force"]).reshape(-1, 4)[0, :3] * e.force_unit
    ref = pair_contact(
        shape, shape, *x, kn=spec.normal_stiffness, kt=spec.tangential_stiffness
    )
    assert np.linalg.norm(force - ref["force"]) / np.linalg.norm(ref["force"]) < 1e-4


def test_drag_energy_accounting(metal, shape):
    spec = Experiment(particles=2, timestep=1e-8)
    x = np.array([[0.001, 0.001, 0.001], [0.004, 0.004, 0.004]])
    e = Engine(ShapeLibrary([shape]), spec, x, [0.006] * 3, [0, 0], [1, 1], metal=metal)
    e.initialize()
    velocity = np.zeros((2, 4), np.float32)
    velocity[:, 0] = 0.1
    e.buffers["velocity"].copy_from_numpy(velocity)
    before = e.diagnostics()
    e.advance(10, drag=100000)
    after = e.diagnostics()
    expected = before["kinetic_energy"] * np.exp(-2 * 100000 * 10 * spec.timestep)
    assert np.isclose(after["kinetic_energy"], expected, rtol=1e-5)
    assert np.isclose(
        after["drag_dissipation"], before["kinetic_energy"] - expected, rtol=1e-4
    )
    assert after["friction_dissipation"] == 0 and after["cell_work"] == 0


def test_contact_growth_from_effective_minimum(metal):
    from sand_atlas_dem.qualification import growth

    result = growth(
        metal,
        sphere(resolution=64, samples=2048),
        {"initial_contacts_per_particle": 1, "relative_state_limit": 1e-4},
    )
    assert result["passed"], result
    assert result["retries"][0]["new_capacities"]["contacts_per_particle"] >= 16


def test_sub_ulp_translation_rotation_and_slow_cell(metal, shape):
    """Analytical motions whose per-step increments round to zero without residuals."""
    from scipy.spatial.transform import Rotation

    spec = Experiment(particles=2, timestep=1e-8)
    x = np.array([[0.001, 0.001, 0.001], [0.004, 0.004, 0.004]])
    e = Engine(ShapeLibrary([shape]), spec, x, [0.006] * 3, [0, 0], [1, 1], metal=metal)
    q = np.tile(np.array([0.5, 0.5, 0.5, 0.5], np.float32), (2, 1))
    e.buffers["rotation"].copy_from_numpy(q)
    e.initialize()
    velocity = np.tile(np.array([1e-7, 0.1, 0, 0], np.float32), (2, 1))
    e.buffers["velocity"].copy_from_numpy(velocity)
    physical_w = np.array([0.01, 0.02, -0.03])
    inertia = shape.inertia * spec.density / (e.mass_unit * e.length_unit**2)
    momentum = inertia @ (physical_w * e.time_unit)
    e.buffers["momentum"].copy_from_numpy(
        np.column_stack([np.tile(momentum, (2, 1)), np.zeros(2)]).astype(np.float32)
    )
    before = e.state()
    steps = 2000
    for _ in range(20):
        e.advance(100, rate=[0, -0.1, 0])
    after = e.state()
    actual_x = after["position"][:, 0].astype(float) + after["positionRemainder"][:, 0]
    expected_x = (
        before["position"][:, 0].astype(float)
        + float(velocity[0, 0]) * steps * spec.timestep / e.time_unit
    )
    assert np.allclose(actual_x, expected_x, rtol=0, atol=5e-11)
    v = after["velocity"][:, 1].astype(float) + after["velocityRemainder"][:, 1]
    assert np.allclose(
        v,
        float(velocity[0, 1]) * np.exp(0.1 * steps * spec.timestep),
        rtol=0,
        atol=1e-10,
    )
    strain = 1 - float(after["cell"][0, 1]) / float(before["cell"][0, 1])
    expected_strain = -np.expm1(-0.1 * steps * spec.timestep)
    assert abs(strain - expected_strain) < 1e-7 and strain > 0.9 * expected_strain
    expected = Rotation.from_rotvec(
        physical_w * steps * spec.timestep
    ) * Rotation.from_quat(q[0])
    actual = Rotation.from_quat(
        after["rotation"][0].astype(float) + after["rotationRemainder"][0]
    )
    assert (actual * expected.inv()).magnitude() < 1e-9
    # Restart must retain the sub-ULP state rather than merely copying visible poses.
    e.advance(100)
    continued = e.state()
    e.restore(after, steps)
    restored = e.state()
    for name in (
        "positionRemainder",
        "velocityRemainder",
        "rotationRemainder",
        "momentumRemainder",
        "cellLog",
    ):
        assert np.array_equal(restored[name], after[name]), name
    e.advance(100)
    replayed = e.state()
    for name in (
        "position",
        "positionRemainder",
        "velocity",
        "velocityRemainder",
        "rotation",
        "rotationRemainder",
        "cellLog",
    ):
        assert np.array_equal(replayed[name], continued[name]), name
