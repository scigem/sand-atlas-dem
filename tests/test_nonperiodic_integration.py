"""Frozen nonperiodic interior mechanics and checkpoint/export regressions."""

import json
import os

import numpy as np
import pytest
import trimesh

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh, sphere
from sand_atlas_dem.gpu import Engine, RejectedStep, f4
from sand_atlas_dem.simulation import Simulation, export_run
from sand_atlas_dem.spec import Experiment

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(
        os.environ.get("SAND_DEM_METAL") != "1", reason="Serial Metal required"
    ),
]


@pytest.mark.parametrize("geometry", ["sdf", "mesh"])
def test_interior_contacts_preserve_existing_mechanics_and_history(geometry):
    shape = from_mesh(
        trimesh.creation.box(extents=[1, 1, 1]), unit="mm", resolution=16, samples=512
    )
    library = ShapeLibrary([shape])
    metal = Metal()
    positions = [[0.002] * 3, [0.002995, 0.002, 0.002]]
    engines = []
    for periodic in (True, False):
        spec = Experiment(
            particles=2, timestep=1e-8, periodic=periodic, contact_geometry=geometry
        )
        engine = Engine(
            library, spec, positions, [0.006] * 3, [0, 0], [1, 1], metal=metal
        )
        engine.initialize()
        v = np.array([[0, 0.01, 0], [0, 0, 0]])
        engine.buffers["velocity"].copy_from_numpy(
            f4(v * engine.time_unit / engine.length_unit)
        )
        engine.advance(3)
        engines.append(engine)
    periodic, bounded = engines
    assert bounded.diagnostics()["active_surface_contacts"] > 0
    for name in ("position", "velocity", "rotation", "momentum", "force", "torque"):
        assert np.allclose(
            periodic.state()[name], bounded.state()[name], rtol=2e-4, atol=3e-7
        ), name
    saved = bounded.state()
    bounded.advance(10)
    expected = bounded.state()
    bounded.restore(saved, 3)
    bounded.advance(10)
    for name in (
        "position",
        "velocity",
        "rotation",
        "momentum",
        "force",
        "torque",
        "accumulated",
    ):
        assert np.allclose(
            expected[name], bounded.state()[name], rtol=2e-4, atol=3e-7
        ), name
    assert bounded.diagnostics()["friction_dissipation"] >= 0


def test_nonperiodic_checkpoint_and_export(tmp_path):
    library = tmp_path / "shapes.h5"
    ShapeLibrary([sphere(resolution=16, samples=64)]).save(library)
    spec = Experiment(
        shape_library=str(library),
        periodic=False,
        particles=27,
        timestep=1e-8,
        initial_solid_fraction=0.05,
        diagnostic_interval=2,
        snapshot_interval=2,
        stages=[dict(kind="hold", steps=8)],
    )
    simulation = Simulation(spec)
    first = simulation.run(tmp_path / "first", max_chunks=2)
    assert first.status == "paused"
    resumed = Simulation.resume(first.path / "checkpoint.h5")
    assert resumed.experiment.periodic is False
    assert resumed.engine.bindings()["c"]["periodic"] == 0
    simulation.engine.advance(2)
    resumed.engine.advance(2)
    expected = simulation.engine.state()
    for name, value in expected.items():
        assert np.array_equal(value, resumed.engine.state()[name]), name
    # Continue laboratory stage accounting from the exact saved checkpoint.
    resumed = Simulation.resume(first.path / "checkpoint.h5")
    last = resumed.run(tmp_path / "last")
    assert last.status == "complete"
    export_run(last.path, tmp_path / "export")
    result = json.loads((tmp_path / "export/atlas-comparison.json").read_text())
    assert result["kind"] == "synthetic-dem"
    assert result["conventions"]["boundary"] == "nonperiodic-centre-domain"
    assert result["conventions"]["measurement_volume"] == "computational_domain"
    assert result["rows"][-1]["step"] == 8


def test_invalid_device_initial_position_is_rejected():
    # Exercise GPU validation even if a caller directly overwrites a device buffer.
    library = ShapeLibrary([sphere(resolution=16, samples=64)])
    spec = Experiment(periodic=False, particles=1, timestep=1e-8)
    engine = Engine(library, spec, [[0.002] * 3], [0.006] * 3, [0], [1])
    engine.buffers["position"].copy_from_numpy(
        f4(np.array([[0.006, 0.002, 0.002]]) / engine.length_unit)
    )
    with pytest.raises(RejectedStep, match="domain escape") as error:
        engine.initialize()
    assert error.value.flags == 2048
