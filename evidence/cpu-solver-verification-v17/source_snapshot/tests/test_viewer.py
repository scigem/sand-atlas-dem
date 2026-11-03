"""Display conversion and rendering checks; no CPU simulation fallback."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from sand_atlas_dem.viewer import ViewOptions, particle_frame, surface_arrays, Renderer
from sand_atlas_dem.util import read_json


def fixture_sim():
    state = {name: np.zeros((2, 4)) for name in ("position", "velocity", "force", "torque", "momentum")}
    state["position"][:, :3] = [[1, 1, 1], [3, 2, 2]]
    state["positionRemainder"] = np.full((2, 4), 1e-8)
    state["velocity"][0, 0] = 2
    state["velocityRemainder"] = np.full((2, 4), 1e-8)
    state["force"][0, 1] = 3
    state["torque"][1, 2] = 4
    state["momentum"][0, 2] = 5
    state["rotation"] = np.tile([0., 0., 0., 1.], (2, 1))
    state["cell"] = np.tile([5., 5., 5., 0.], (3, 1))
    state["meta"] = np.array([[0, 1, 1, 0], [1, 0, 2, 0], [0, 1, 3, 0]])
    engine = SimpleNamespace(state=lambda: state, ids=np.array([0, 0]), scales=np.array([1., 2.]),
                             length_unit=2., time_unit=4., force_unit=10., mass_unit=3.,
                             step=0, spec=SimpleNamespace(timestep=0.1))
    shape = SimpleNamespace(vertices=np.array([[0., 0., 0.], [0.3, 0., 0.], [0., 0.3, 0.], [0., 0., 0.3]]),
                            faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]), radius=0.3)
    return SimpleNamespace(engine=engine, library=SimpleNamespace(shapes=[shape]))


def test_si_units_compensation_and_unique_coordination():
    sim = fixture_sim()
    frame = particle_frame(sim.engine)
    np.testing.assert_allclose(frame["position"][0], np.full(3, 2 + 2e-8), rtol=0, atol=1e-14)
    assert frame["velocity"][0, 0] == pytest.approx(1 + 5e-9)
    assert frame["force"][0, 1] == 30
    assert frame["torque"][1, 2] == 80
    assert frame["momentum"][0, 2] == 15
    np.testing.assert_array_equal(frame["coordination"], [1, 1])
    sim.engine.state()["meta"] = np.empty((0, 4), dtype=int)
    np.testing.assert_array_equal(particle_frame(sim.engine)["coordination"], [0, 0])


def test_mesh_pose_scale_and_topology():
    sim = fixture_sim()
    frame = particle_frame(sim.engine)
    frame["rotation"][1] = [0, 0, 1, 0]  # 180 degrees about z
    points, faces, owners = surface_arrays(sim.library, sim.engine.ids, sim.engine.scales, frame)
    np.testing.assert_allclose(points[5], frame["position"][1] + [-0.6, 0, 0])
    assert faces.reshape(-1, 4)[4].tolist() == [3, 4, 6, 5]
    np.testing.assert_array_equal(owners, [0] * 4 + [1] * 4)


@pytest.mark.parametrize("kwargs", [{"every": 0}, {"fps": 0}, {"limit": float("nan")}, {"limit": -1}, {"max_vectors": 0}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        ViewOptions(**kwargs)


def test_render_failure_does_not_advance_or_modify_solver(tmp_path, monkeypatch):
    pytest.importorskip("pyvista")
    sim = fixture_sim()
    viewer = Renderer(sim, tmp_path, ViewOptions())
    monkeypatch.setattr(viewer, "_draw", lambda frame: (_ for _ in ()).throw(RuntimeError("display lost")))
    with pytest.warns(UserWarning, match="simulation continues"):
        viewer.update()
    assert sim.engine.step == 0
    assert viewer.close()["status"] == "failed"
    assert "display lost" in read_json(tmp_path / "visualization/render.json")["failure"]


@pytest.mark.render
@pytest.mark.skipif(os.environ.get("SAND_DEM_RENDER") != "1", reason="Explicit graphical-session rendering required")
@pytest.mark.parametrize("geometry", ["mesh", "sphere"])
def test_four_real_mp4_views_and_final_frame(tmp_path, geometry):
    pytest.importorskip("pyvista")
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    sim = fixture_sim()
    options = ViewOptions(video=True, every=3, field="force", vectors="force", geometry=geometry)
    viewer = Renderer(sim, tmp_path, options)
    viewer.update()
    sim.engine.step = 1
    viewer.update()  # skipped
    sim.engine.step = 2
    viewer.update(final=True)
    viewer.update(final=True)  # no duplicate endpoint
    report = viewer.close()
    assert report["failure"] is None, report
    assert report["frames"] == 2
    frames = []
    for view in ("x", "y", "z", "iso"):
        with imageio.get_reader(tmp_path / f"visualization/{view}.mp4") as reader:
            assert reader.count_frames() == 2
            frame = reader.get_data(0)
            assert frame.shape == (720, 960, 3)
            assert frame.std() > 10
            frames.append(frame)
    assert not np.array_equal(frames[0], frames[1])


@pytest.mark.render
@pytest.mark.skipif(os.environ.get("SAND_DEM_RENDER") != "1", reason="Explicit graphical-session rendering required")
def test_live_window_close_continues_recording(tmp_path):
    pytest.importorskip("pyvista")
    pytest.importorskip("imageio_ffmpeg")
    sim = fixture_sim()
    viewer = Renderer(sim, tmp_path, ViewOptions(live=True, video=True, every=2))
    viewer.update()
    viewer.live.close()
    sim.engine.step = 1
    viewer.update()
    sim.engine.step = 2
    viewer.update(final=True)
    report = viewer.close()
    assert report["failure"] is None, report
    assert report["frames"] == 2
    assert (tmp_path / "visualization/iso.mp4").stat().st_size > 1000
