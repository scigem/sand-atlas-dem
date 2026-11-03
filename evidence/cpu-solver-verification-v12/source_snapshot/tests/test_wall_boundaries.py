"""Prospective plane integration checks; gates frozen before Metal execution."""

import copy
import json
import os

import h5py
import numpy as np
import pytest
import trimesh

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.boundaries import BOX_NORMALS, box_extents
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh, sphere
from sand_atlas_dem.gpu import Engine, RejectedStep, f4
from sand_atlas_dem.simulation import Simulation, export_run
from sand_atlas_dem.spec import Experiment

metal_test = pytest.mark.skipif(
    os.environ.get("SAND_DEM_METAL") != "1", reason="Serial Metal required"
)


def plane(normal=(0, 0, 1), offset=0.00155, velocity=(0, 0, 0)):
    return dict(
        normal=list(normal),
        offset=offset,
        velocity=list(velocity),
        normal_stiffness=1e12,
        tangential_stiffness=5e11,
        friction=0.3,
    )


def box_planes(low=0.0005, high=0.0055):
    return [
        plane(n, low if i % 2 == 0 else -high)
        for i, n in enumerate(BOX_NORMALS.tolist())
    ]


def test_wall_input_contract():
    with pytest.raises(ValueError, match="nonperiodic"):
        Experiment(planes=[plane()])
    for change in (
        dict(normal=[0, 0, 2]),
        dict(offset=float("nan")),
        dict(velocity=[0, 0]),
        dict(normal_stiffness=0),
        dict(tangential_stiffness=-1),
        dict(friction=-1),
        dict(extra=0),
    ):
        value = plane()
        value.update(change)
        with pytest.raises(ValueError):
            Experiment(periodic=False, planes=[value])
    for ids in ([0] * 6, [0, 1, 2, 3, 4, 6], list(range(5)), [0, 1, 2, 3, 5, 4]):
        with pytest.raises(ValueError):
            Experiment(periodic=False, planes=box_planes(), boundary_box=ids)
    assert np.allclose(
        box_extents(np.array([p["offset"] for p in box_planes()]), list(range(6))),
        0.005,
    )
    with pytest.raises(ValueError, match="nonpositive"):
        Experiment(
            periodic=False, planes=box_planes(low=0.006), boundary_box=list(range(6))
        )
    with pytest.raises(ValueError, match="planes"):
        Experiment(
            periodic=False, stages=[dict(kind="wall", steps=2, wall_velocities=[])]
        )
    value = Experiment(
        periodic=False,
        planes=[plane()],
        stages=[dict(kind="wall", steps=2, wall_velocities=[[0, 0, 0.1]])],
    )
    assert Experiment(**value.to_dict()).planes == value.planes


@pytest.fixture(scope="module")
def metal():
    return Metal()


@pytest.fixture(scope="module")
def cap():
    return sphere(resolution=16, samples=32768)


def cap_engine(metal, cap, **settings):
    spec = Experiment(
        periodic=False,
        particles=1,
        timestep=1e-8,
        max_overlap_fraction=0.1,
        contacts_per_particle=4096,
        planes=[plane()],
        **settings,
    )
    return Engine(
        ShapeLibrary([cap]), spec, [[0.002] * 3], [0.006] * 3, [0], [1], metal=metal
    )


@pytest.mark.metal
@metal_test
def test_full_area_cap_force_reaction_moment_and_energy(metal, cap):
    engine = cap_engine(metal, cap)
    engine.initialize()
    row = engine.diagnostics()
    radius, depth = 0.0005, 0.00005
    force = np.pi * 1e12 * radius * depth**2
    energy = np.pi * 1e12 * radius * depth**3 / 3
    state = engine.state()
    actual = state["force"][0, :3] * engine.force_unit
    assert abs(actual[2] / force - 1) < 0.001
    assert abs(row["elastic_energy_proxy"] / energy - 1) < 0.001
    reaction = np.array(row["walls"][0]["reaction"])
    assert np.linalg.norm(actual + reaction) < force * 1e-5
    torque = state["torque"][0, :3] * engine.energy_unit
    origin_moment = np.cross([0.002] * 3, actual) + torque
    assert (
        np.linalg.norm(origin_moment + row["walls"][0]["reaction_moment_origin"])
        < force * 0.006 * 1e-5
    )
    assert (
        row["walls"][0]["active_surface_contacts"] == row["active_surface_contacts"] > 0
    )
    assert row["active_grain_pairs"] == 0
    assert row["wall_work"] == row["walls"][0]["work"] == 0
    # Independent common-node virial, without a half-weight wall factor.
    nodes = cap.samples + [0.002] * 3
    depths = np.maximum(0, 0.00155 - nodes[:, 2])
    nodal_force = 1e12 * cap.areas * depths
    expected_stress = (
        -np.outer([0, 0, 1], np.sum(nodal_force[:, None] * cap.samples, axis=0))
        / 0.006**3
    )
    assert (
        np.linalg.norm(np.array(row["contact_stress"]) - expected_stress)
        < force * radius / 0.006**3 * 1e-4
    )


@pytest.mark.metal
@metal_test
@pytest.mark.parametrize("geometry", ["sdf", "mesh"])
def test_rotated_mesh_wall_collection_matches_exhaustive_original_nodes(
    metal, geometry
):
    from scipy.spatial.transform import Rotation

    shape = from_mesh(
        trimesh.creation.box(extents=[1, 0.8, 0.6]),
        unit="mm",
        resolution=16,
        samples=4096,
    )
    normal = np.array([1, 2, 3], float) / np.sqrt(14)
    rotation = Rotation.from_euler("xyz", [0.2, -0.3, 0.4])
    center = np.array([0.002] * 3)
    arms = rotation.apply(shape.samples)
    offset = (arms @ normal).min() + center @ normal + 0.00001
    spec = Experiment(
        periodic=False,
        contact_geometry=geometry,
        particles=1,
        timestep=1e-9,
        max_overlap_fraction=0.03,
        contacts_per_particle=4096,
        planes=[plane(normal, offset)],
    )
    engine = Engine(
        ShapeLibrary([shape]), spec, [center], [0.006] * 3, [0], [1], metal=metal
    )
    engine.buffers["rotation"].copy_from_numpy(
        rotation.as_quat().astype(np.float32).reshape(1, 4)
    )
    engine.initialize()
    depth = np.maximum(0, offset - (arms + center) @ normal)
    force = shape.areas[:, None] * depth[:, None] * 1e12 * normal
    expected_force, expected_torque = force.sum(0), np.cross(arms, force).sum(0)
    state, row = engine.state(), engine.diagnostics()
    assert row["active_surface_contacts"] == np.count_nonzero(depth)
    assert set(state["meta"][:, 2]) == set(np.flatnonzero(depth))
    assert (
        np.linalg.norm(state["force"][0, :3] * engine.force_unit - expected_force)
        < np.linalg.norm(expected_force) * 1e-4
    )
    assert (
        np.linalg.norm(state["torque"][0, :3] * engine.energy_unit - expected_torque)
        < np.linalg.norm(expected_force) * shape.radius * 1e-4
    )


@pytest.mark.metal
@metal_test
def test_moving_wall_work_balance_and_velocity_change(metal, cap):
    engine = cap_engine(metal, cap)
    engine.initialize()
    initial = engine.diagnostics()
    energy0 = initial["elastic_energy_proxy"]
    expected_work = 0
    previous = initial
    for velocity in ([0, 0, 0.1], [0, 0, -0.1]):
        for _ in range(50):
            engine.advance(1, wall_velocities=[velocity])
            row = engine.diagnostics()
            # Independent SI actuator work using measured opposite reactions.
            mean_force = -0.5 * (
                np.array(previous["walls"][0]["reaction"]) + row["walls"][0]["reaction"]
            )
            expected_work += np.dot(mean_force, velocity) * engine.spec.timestep
            assert abs(row["wall_work"] - expected_work) < energy0 * 1e-5
            assert abs(row["walls"][0]["work"] - row["wall_work"]) < energy0 * 1e-5
            residual = row["wall_work"] - (
                row["kinetic_energy"] + row["elastic_energy_proxy"] - energy0
            )
            assert abs(residual) < energy0 * 0.001
            assert row["cell_work"] == 0
            previous = row
    assert abs(previous["walls"][0]["offset"] - initial["walls"][0]["offset"]) < 1e-11


@pytest.mark.metal
@metal_test
def test_wall_friction_separation_return_and_restart(metal, cap):
    engine = cap_engine(metal, cap)
    engine.initialize()
    engine.advance(3, wall_velocities=[[0.1, 0, 0]])
    saved = engine.state()
    assert np.max(np.abs(saved["slip"][:, :3])) > 0
    engine.advance(5)
    expected = engine.state()
    engine.restore(saved, 3)
    engine.advance(5)
    for name in (
        "position",
        "velocity",
        "rotation",
        "momentum",
        "force",
        "torque",
        "accumulated",
        "wallPlane",
        "wallRemainder",
        "wallVelocity",
        "wallAccumulated",
    ):
        assert np.allclose(
            expected[name], engine.state()[name], rtol=2e-4, atol=3e-7
        ), name
    # Independent new initial value state: remove and return the plane at dt=0.
    separated = copy.deepcopy(engine.state())
    separated["wallPlane"][0, 3] = 0
    separated["wallRemainder"][0, 3] = 0
    engine.restore(separated, 8)
    assert engine.state()["contacts"] == 0
    returned = copy.deepcopy(engine.state())
    returned["wallPlane"] = saved["wallPlane"]
    returned["wallRemainder"] = saved["wallRemainder"]
    engine.restore(returned, 8)
    assert engine.state()["contacts"] > 0
    assert np.array_equal(
        engine.state()["slip"][:, :3], np.zeros((engine.state()["contacts"], 3))
    )


@pytest.mark.metal
@metal_test
def test_wall_motion_overlap_failure_rolls_back_velocity_history_and_work(metal, cap):
    engine = cap_engine(metal, cap)
    engine.initialize()
    engine.advance(2, wall_velocities=[[0.1, 0, 0]])
    before = engine.state()
    with pytest.raises(RejectedStep, match="penetration"):
        engine.advance(3, wall_velocities=[[0, 0, 1e5]])
    for name, value in before.items():
        assert np.array_equal(engine.state()[name], value), name
    assert engine.step == 2
    engine.advance(1)


@pytest.mark.metal
@metal_test
def test_wall_contact_growth_retries_without_truncation(metal, cap):
    # Initialise with no active plane nodes, then approach into contact. Compare
    # automatic growth with an otherwise identical generously allocated run.
    results = []
    for capacity in (1, 4096):
        wall = plane(offset=0.0014999)
        spec = Experiment(
            periodic=False,
            particles=1,
            timestep=1e-8,
            max_overlap_fraction=0.1,
            contacts_per_particle=capacity,
            planes=[wall],
        )
        engine = Engine(
            ShapeLibrary([cap]), spec, [[0.002] * 3], [0.006] * 3, [0], [1], metal=metal
        )
        engine.initialize()
        engine.advance(2, wall_velocities=[[0, 0, 100]])
        results.append(engine)
    grown, baseline = results
    assert grown.capacity_retries
    assert grown.state()["contacts"] == baseline.state()["contacts"] > 16
    for name in (
        "position",
        "velocity",
        "force",
        "accumulated",
        "wallPlane",
        "wallAccumulated",
    ):
        assert np.allclose(
            grown.state()[name], baseline.state()[name], rtol=2e-4, atol=3e-7
        ), name


@pytest.mark.metal
@metal_test
def test_wall_box_volume_strain_and_invalid_gap_rollback(metal):
    shape = sphere(resolution=16, samples=64)
    spec = Experiment(
        periodic=False,
        particles=1,
        planes=box_planes(),
        boundary_box=list(range(6)),
        timestep=1e-8,
    )
    engine = Engine(
        ShapeLibrary([shape]), spec, [[0.003] * 3], [0.006] * 3, [0], [1], metal=metal
    )
    engine.initialize()
    velocities = np.zeros((6, 3))
    velocities[5, 2] = -1
    engine.advance(10, wall_velocities=velocities)
    row = engine.diagnostics()
    assert np.allclose(row["domain_cell"], [0.006] * 3, rtol=1e-7)
    assert np.allclose(row["cell"], [0.005, 0.005, 0.005 - 1e-7], rtol=1e-7)
    assert abs(row["axial_strain"] - 2e-5) < 1e-7
    before = engine.state()
    velocities[5, 2] = -1e6
    with pytest.raises(RejectedStep, match="invalid boundary box"):
        engine.advance(1, wall_velocities=velocities)
    for name, value in before.items():
        assert np.array_equal(value, engine.state()[name]), name


@pytest.mark.metal
@metal_test
def test_wall_hdf5_checkpoint_stage_schedule_and_export(tmp_path, metal):
    library = tmp_path / "shape.h5"
    ShapeLibrary([sphere(resolution=16, samples=64)]).save(library)
    velocities = np.zeros((6, 3))
    velocities[5, 2] = -0.1
    spec = Experiment(
        shape_library=str(library),
        periodic=False,
        particles=1,
        cell=[0.006] * 3,
        planes=box_planes(),
        boundary_box=list(range(6)),
        timestep=1e-8,
        diagnostic_interval=2,
        snapshot_interval=2,
        stages=[dict(kind="wall", steps=8, wall_velocities=velocities.tolist())],
    )
    simulation = Simulation(spec, metal=metal)
    first = simulation.run(tmp_path / "first", max_chunks=2)
    assert first.status == "paused"
    with h5py.File(first.path / "checkpoint.h5") as f:
        assert f.attrs["format"] == "sand-dem-checkpoint/4"
        assert "wallPlane" in f and "wallAccumulated" in f
    resumed = Simulation.resume(first.path / "checkpoint.h5", metal=metal)
    last = resumed.run(tmp_path / "last")
    assert last.status == "complete"
    assert last.rows[-1]["walls"][5]["offset"] > first.rows[0]["walls"][5]["offset"]
    export_run(last.path, tmp_path / "export")
    result = json.loads((tmp_path / "export/atlas-comparison.json").read_text())
    assert result["conventions"]["boundary"] == "translating-planes"
    assert result["conventions"]["measurement_volume"] == "orthogonal-wall-box"
    assert result["rows"][-1]["step"] == 8


@pytest.mark.metal
@metal_test
def test_restart_rejects_changed_normal_and_unrepresentable_velocity_before_mutation(
    metal, cap
):
    engine = cap_engine(metal, cap)
    engine.initialize()
    before = engine.state()
    changed = copy.deepcopy(before)
    changed["wallPlane"][0, :3] = [0, 1, 0]
    with pytest.raises(ValueError, match="normal identity"):
        engine.restore(changed, 0)
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="FP32"):
            engine.advance(1, wall_velocities=[[1e300, 0, 0]])
    for name, value in before.items():
        assert np.array_equal(value, engine.state()[name]), name
