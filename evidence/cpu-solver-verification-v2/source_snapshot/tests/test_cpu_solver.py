"""Explicit CPU solver contracts. No GPU is created or substituted in these tests.

Frozen oracle tolerances: force/torque 2e-4 relative for normalized FP32 mesh/SDF;
absolute SI force 1e-9 N, torque 1e-13 Nm; replay state 2e-6 relative, 1e-9
normalized absolute. Exact six-node cube geometry is used for closed-form checks.
"""

from dataclasses import replace
import json

import numpy as np
import pytest
import trimesh

from sand_atlas_dem.cpu import CPU
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.gpu import Engine, RejectedStep, f4
from sand_atlas_dem.reference import pair_contact
from sand_atlas_dem.simulation import Simulation, export_run
from sand_atlas_dem.spec import Experiment


@pytest.fixture(scope="module")
def cpu():
    return CPU()


@pytest.fixture(scope="module")
def cube():
    h = 1e-4
    shape = from_mesh(
        trimesh.creation.box(extents=[h] * 3),
        unit="m",
        resolution=16,
        samples=64,
        provenance={"source": "synthetic-exact-cube", "units": "m"},
    )
    shape.samples = np.concatenate([np.eye(3), -np.eye(3)]) * h / 2
    shape.areas = np.full(6, h * h)
    shape.validate()
    return shape


def make(
    cpu,
    cube,
    *,
    geometry="mesh",
    viscosity=0,
    policy="active-contact",
    friction=0.5,
    source=False,
    wall=False,
    dt=1e-9,
    separation=9.99e-5,
):
    planes = (
        [
            dict(
                normal=[0, 0, 1],
                offset=0.005 - 5e-5 + 1e-7,
                velocity=[0, 0, 0],
                normal_stiffness=1e12,
                tangential_stiffness=5e11,
                friction=friction,
                normal_viscosity=viscosity,
            )
        ]
        if wall
        else []
    )
    positions = (
        [[0.005] * 3] if wall else [[0.005] * 3, [0.005 + separation, 0.005, 0.005]]
    )
    spec = Experiment(
        particles=len(positions),
        periodic=not wall,
        planes=planes,
        contact_geometry=geometry,
        timestep=dt,
        timestep_policy=policy,
        normal_viscosity=viscosity,
        friction=friction,
        max_overlap_fraction=0.1,
        contact_normal="source-surface-experimental" if source else "target-gradient",
        contacts_per_particle=128,
    )
    return Engine(
        ShapeLibrary([cube]),
        spec,
        positions,
        [0.01] * 3,
        [0] * len(positions),
        [1] * len(positions),
        metal=cpu,
    )


@pytest.mark.parametrize("geometry", ["mesh", "sdf"])
def test_pair_force_torque_against_independent_reference(cpu, cube, geometry):
    e = make(cpu, cube, geometry=geometry)
    e.initialize()
    if geometry == "mesh":
        expected_force = np.array([-0.001, 0, 0])
        expected_torque = np.zeros(3)
    else:
        result = pair_contact(
            cube,
            cube,
            [0.005] * 3,
            [0.005 + 9.99e-5, 0.005, 0.005],
            kn=1e12,
            kt=5e11,
            mu=0.5,
        )
        expected_force = result["force"]
        expected_torque = result["torque_i"]
    force = cpu.read(e.buffers["force"], shape=(2, 4))[:, :3] * e.force_unit
    torque = cpu.read(e.buffers["torque"], shape=(2, 4))[:, :3] * e.energy_unit
    np.testing.assert_allclose(force[0], expected_force, rtol=2e-4, atol=1e-9)
    np.testing.assert_allclose(force[1], -expected_force, rtol=2e-4, atol=1e-9)
    np.testing.assert_allclose(torque[0], expected_torque, rtol=2e-4, atol=1e-13)
    e.advance(10)
    row = e.diagnostics()
    assert row["step"] == 10 and row["active_grain_pairs"] == 1
    assert row["kinetic_energy"] > 0 and row["elastic_energy_proxy"] > 0
    assert np.isfinite(row["stress"]).all()
    np.testing.assert_allclose(
        np.sum(cpu.read(e.buffers["velocity"], shape=(2, 4))[:, :3], axis=0),
        0,
        atol=1e-8,
    )


@pytest.mark.parametrize("geometry", ["sdf", "mesh"])
def test_friction_history_and_restart(cpu, cube, geometry):
    e = make(cpu, cube, geometry=geometry)
    velocity = np.array([[0, 0.01, 0], [0, -0.01, 0]]) * e.time_unit / e.length_unit
    e.buffers["velocity"].copy_from_numpy(f4(velocity))
    e.initialize()
    e.advance(7)
    state = e.state()
    assert np.linalg.norm(state["slip"][:, :3]) > 0
    e.advance(5)
    expected = e.state()
    restored = make(cpu, cube, geometry=geometry)
    restored.restore(state, 7)
    restored.advance(5)
    actual = restored.state()
    for name in [
        "position",
        "velocity",
        "rotation",
        "momentum",
        "force",
        "torque",
        "slip",
        "accumulated",
    ]:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-6, atol=1e-9)
    np.testing.assert_array_equal(actual["meta"], expected["meta"])


@pytest.mark.parametrize("viscosity", [0, 1e6])
def test_wall_force_motion_loss_and_restart(cpu, cube, viscosity):
    e = make(cpu, cube, wall=True, viscosity=viscosity)
    velocity = np.array([[0.01, 0, -0.01]]) * e.time_unit / e.length_unit
    e.buffers["velocity"].copy_from_numpy(f4(velocity))
    e.initialize()
    row = e.diagnostics()
    expected = 0.001 + viscosity * 1e-8 * 0.01
    force = cpu.read(e.buffers["force"], shape=(1, 4))[0, :3] * e.force_unit
    assert force[2] == pytest.approx(expected, rel=2e-4, abs=1e-9)
    np.testing.assert_allclose(
        row["walls"][0]["reaction"], -force, rtol=2e-4, atol=1e-9
    )
    e.advance(3, wall_velocities=[[0, 0, 0.001]])
    state = e.state()
    e.advance(2)
    expected = e.state()
    restored = make(cpu, cube, wall=True, viscosity=viscosity)
    restored.restore(state, 3)
    restored.advance(2)
    for name in [
        "position",
        "velocity",
        "force",
        "normalDamping",
        "wallPlane",
        "wallRemainder",
        "wallAccumulated",
    ]:
        if name in expected:
            np.testing.assert_allclose(
                restored.state()[name], expected[name], rtol=2e-6, atol=1e-9
            )
    final = e.diagnostics()
    assert final["normal_contact_dissipation"] >= 0
    assert final["walls"][0]["offset"] > row["walls"][0]["offset"]


@pytest.mark.parametrize("policy", ["conservative", "active-contact"])
def test_timestep_rejection_rolls_back_batch(cpu, cube, policy):
    e = make(cpu, cube, policy=policy)
    e.initialize()
    before = e.state()
    with pytest.raises(RejectedStep):
        e.advance(2, rate=[-1e8, 0, 0])
    after = e.state()
    assert e.step == 0
    for key in before:
        if isinstance(before[key], np.ndarray):
            np.testing.assert_array_equal(after[key], before[key])


def test_conservative_timestep_preflight(cpu, cube):
    with pytest.raises(ValueError, match="conservative estimate"):
        make(cpu, cube, policy="conservative", dt=0.01)


def test_active_timestep_rejection(cpu, cube):
    e = make(cpu, cube)
    e.initialize()
    e.spec = replace(e.spec, timestep=0.01)
    with pytest.raises(RejectedStep):
        e.advance(2)
    assert e.step == 0


def test_source_normals_explicit_cpu_path(cpu, cube):
    e = make(cpu, cube, source=True, friction=0)
    e.initialize()
    e.advance(2)
    assert e.diagnostics()["contact_normal"] == "source-surface-experimental"


def test_strain_cell_and_drag(cpu, cube):
    e = make(cpu, cube, separation=0.0002)
    e.buffers["velocity"].copy_from_numpy(
        f4(np.array([[0.01, 0, 0], [-0.01, 0, 0]]) * e.time_unit / e.length_unit)
    )
    e.initialize()
    initial = e.diagnostics()["kinetic_energy"]
    e.advance(10, rate=[-1, 0, 0], drag=1e5)
    row = e.diagnostics()
    assert row["cell"][0] < 0.01 and row["kinetic_energy"] < initial
    assert row["drag_dissipation"] > 0


def test_hierarchical_scan_matches_exclusive_sum():
    from sand_atlas_dem.gpu import Scan

    cpu = CPU()
    values = np.arange(777, dtype=np.uint32) % 17
    source = cpu.buffer(values)
    dest = cpu.buffer(count=len(values))
    encoder = cpu.create_command_encoder()
    Scan(cpu, len(values)).encode(encoder, source, dest)
    cpu.submit(encoder)
    np.testing.assert_array_equal(cpu.read(dest, np.uint32), np.cumsum(values) - values)


def test_complete_simulation_pause_resume_export(cpu, cube, tmp_path):
    library = ShapeLibrary([cube])
    library_path = tmp_path / "cube.h5"
    library.save(library_path)
    spec = Experiment(
        shape_library=str(library_path),
        particles=8,
        timestep=1e-9,
        contact_geometry="mesh",
        diagnostic_interval=2,
        snapshot_interval=2,
        stages=[dict(kind="strain", steps=6, rate=[-0.1, -0.1, -0.1], drag=0)],
    )
    sim = Simulation(spec, backend="cpu")
    partial = sim.run(tmp_path / "partial", max_chunks=1)
    assert partial.status == "paused"
    resumed = Simulation.resume(tmp_path / "partial/checkpoint.h5")
    assert resumed.engine.m.name == "cpu"
    result = resumed.run(tmp_path / "complete")
    assert result.status == "complete" and resumed.engine.step == 6
    export_run(result.path, tmp_path / "exports")
    manifest = json.loads((result.path / "manifest.json").read_text())
    assert (
        manifest["backend"]["backend"] == "cpu" and not manifest["scientific_admission"]
    )
    assert (tmp_path / "exports/atlas-comparison.json").exists()
    continuous = Simulation(spec, backend="cpu")
    continuous.run(tmp_path / "continuous")
    np.testing.assert_array_equal(
        resumed.engine.state()["position"], continuous.engine.state()["position"]
    )


def test_mesh_query_concavity_and_packed_shape_identity(cpu):
    from pathlib import Path
    from sand_atlas_dem.reference import triangle_nearest_points

    library = ShapeLibrary.load(Path("assets/generated/concave-prism.h5"))
    shape = library.shapes[0]
    e = Engine(
        library,
        Experiment(particles=2, contact_geometry="mesh", timestep=1e-9),
        [[0.002] * 3, [0.005] * 3],
        [0.01] * 3,
        [0, 0],
        [1, 1],
        metal=cpu,
    )
    rng = np.random.default_rng(93)
    points = rng.uniform(
        shape.vertices.min(0) * 1.2, shape.vertices.max(0) * 1.2, (257, 3)
    )
    # Normalize the independent oracle too: Trimesh's degeneracy tolerances are
    # absolute and can otherwise misclassify these SI-scale triangles.
    mesh = trimesh.Trimesh(shape.vertices / e.length_unit, shape.faces, process=False)
    _, distance, _ = triangle_nearest_points(mesh, points / e.length_unit)
    expected = (
        np.where(mesh.contains(points / e.length_unit), -distance, distance)
        * e.length_unit
    )
    source = cpu.buffer(f4(points / e.length_unit))
    result = cpu.buffer(count=len(points), stride=16)
    cpu.dispatch(
        "meshProbe",
        len(points),
        r=e.bindings(),
        count=len(points),
        shape=0,
        points=source,
        result=result,
    )
    values = cpu.read(result, shape=(len(points), 4))
    np.testing.assert_allclose(
        values[:, 3] * e.length_unit, expected, atol=2e-9, rtol=2e-4
    )
    np.testing.assert_array_equal(values[:, 3] < 0, expected < 0)


def test_periodic_contact_uses_unique_minimum_image(cpu, cube):
    e = make(cpu, cube)
    positions = np.array([[0.00004995, 0.005, 0.005], [0.00995005, 0.005, 0.005]])
    e.buffers["position"].copy_from_numpy(f4(positions / e.length_unit))
    e.buffers["positionRemainder"].copy_from_numpy(
        f4(positions / e.length_unit - (positions / e.length_unit).astype(np.float32))
    )
    e.initialize()
    force = cpu.read(e.buffers["force"], shape=(2, 4))[:, :3] * e.force_unit
    # The normalized FP32 cell itself has a representational error. Verify the
    # force for the actual stored state, independently reconstructing its branch.
    state = e.state()
    p = state["position"][:, :3].astype(float) + state["positionRemainder"][
        :, :3
    ].astype(float)
    cell = state["cell"][0, :3].astype(float)
    branch = p[1] - p[0]
    branch -= np.floor(branch / cell + 0.5) * cell
    magnitude = 1e12 * 1e-8 * (1e-4 - abs(branch[0] * e.length_unit))
    np.testing.assert_allclose(
        force, [[magnitude, 0, 0], [-magnitude, 0, 0]], rtol=2e-4, atol=1e-9
    )
    assert e.diagnostics()["candidate_pairs"] == 1


def test_contact_capacity_growth_is_explicit_and_replays(cpu):
    shape = from_mesh(trimesh.creation.box(), unit="mm", resolution=8, samples=1024)
    spec = Experiment(
        particles=2,
        timestep=1e-9,
        contact_geometry="mesh",
        contacts_per_particle=16,
        pairs_per_particle=4,
    )
    e = Engine(
        ShapeLibrary([shape]),
        spec,
        [[0.005] * 3, [0.00602, 0.005, 0.005]],
        [0.02] * 3,
        [0, 0],
        [1, 1],
        metal=cpu,
    )
    e.initialize()
    capacity = e.contact_capacity
    velocity = np.zeros((2, 4), np.float32)
    velocity[1, 0] = -0.00003 / spec.timestep * e.time_unit / e.length_unit
    e.buffers["velocity"].copy_from_numpy(velocity)
    e.advance(1)
    assert e.state()["contacts"] > capacity
    assert e.contact_capacity >= e.state()["contacts"] and e.capacity_retries
    assert e.m.name == "cpu"


def test_multilevel_wall_reduction_counts_force_and_moment(cpu, cube):
    n = 257
    centers = np.array(
        [
            [0.002 + (i % 17) * 0.0002, 0.002 + (i // 17) * 0.0002, 0.005]
            for i in range(n)
        ]
    )
    planes = [
        dict(
            normal=[0, 0, 1],
            offset=0.005 - 5e-5 + 1e-7,
            velocity=[0, 0, 0],
            normal_stiffness=1e12,
            tangential_stiffness=5e11,
            friction=0.5,
        ),
        dict(
            normal=[0, 0, -1],
            offset=-(0.005 + 5e-5 - 1e-7),
            velocity=[0, 0, 0],
            normal_stiffness=1e12,
            tangential_stiffness=5e11,
            friction=0.5,
        ),
    ]
    spec = Experiment(
        periodic=False,
        planes=planes,
        particles=n,
        timestep=1e-9,
        contact_geometry="mesh",
        max_overlap_fraction=0.1,
    )
    e = Engine(
        ShapeLibrary([cube]), spec, centers, [0.01] * 3, [0] * n, [1] * n, metal=cpu
    )
    e.initialize()
    row = e.diagnostics()
    assert len(e.wall_diag_buffers) == 2
    for normal, wall in zip([np.array([0, 0, 1]), np.array([0, 0, -1])], row["walls"]):
        reaction = -0.001 * n * normal
        moment = -np.cross(centers, np.tile(0.001 * normal, (n, 1))).sum(0)
        assert wall["active_surface_contacts"] == n
        np.testing.assert_allclose(wall["reaction"], reaction, rtol=2e-4, atol=1e-9)
        np.testing.assert_allclose(
            wall["reaction_moment_origin"], moment, rtol=2e-4, atol=1e-12
        )
