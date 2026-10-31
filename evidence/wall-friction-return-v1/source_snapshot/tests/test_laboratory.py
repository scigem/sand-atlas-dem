import os

import pytest

from sand_atlas_dem.geometry import ShapeLibrary, sphere
from sand_atlas_dem.simulation import Simulation, export_run
from sand_atlas_dem.spec import Experiment

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(
        os.environ.get("SAND_DEM_METAL") != "1",
        reason="Explicit Metal execution required",
    ),
]


def test_run_resume_and_export(tmp_path):
    library = tmp_path / "shapes.h5"
    ShapeLibrary([sphere(resolution=16, samples=64)]).save(library)
    spec = Experiment(
        shape_library=str(library),
        particles=27,
        timestep=1e-8,
        initial_solid_fraction=0.05,
        diagnostic_interval=2,
        snapshot_interval=2,
        stages=[dict(kind="hold", steps=8)],
    )
    sim = Simulation(spec)
    first = sim.run(tmp_path / "first", max_chunks=2)
    assert first.status == "paused"
    resumed = Simulation.resume(first.path / "checkpoint.h5")
    last = resumed.run(tmp_path / "resumed")
    assert last.status == "complete" and last.rows[-1]["step"] == 8
    export_run(last.path, tmp_path / "export")
    assert (tmp_path / "export" / "atlas-comparison.json").exists()
    # Source hashing rejects modified shape assets rather than silently restarting.
    with open(library, "ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        Simulation.resume(first.path / "checkpoint.h5")


def test_unconverged_preparation_is_failure(tmp_path):
    library = tmp_path / "s.h5"
    ShapeLibrary([sphere(resolution=16, samples=64)]).save(library)
    spec = Experiment(
        shape_library=str(library),
        particles=27,
        timestep=1e-8,
        initial_solid_fraction=0.05,
        diagnostic_interval=1,
        stages=[
            dict(
                kind="isotropic",
                steps=2,
                drag=1000,
                pressure=1000,
                gain=1,
                filter_time=1e-5,
                max_rate=1,
                converged_samples=100,
            )
        ],
    )
    result = Simulation(spec).run(tmp_path / "failed")
    assert result.status == "failed"
