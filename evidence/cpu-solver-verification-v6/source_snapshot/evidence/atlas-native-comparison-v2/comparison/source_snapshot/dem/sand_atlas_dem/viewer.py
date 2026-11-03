"""Optional rendering of committed DEM state. No solver or GPU kernel changes."""

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

from .util import quat_matrix, write_json

FIELDS = {
    "speed": "Peculiar speed (m/s)",
    "force": "Net contact force (N)",
    "torque": "Net contact torque (N m)",
    "momentum": "Angular momentum (kg m²/s)",
    "coordination": "Active grain neighbours",
    "shape": "Shape index",
}
VECTORS = ("none", "velocity", "force", "torque", "momentum")
VIEWS = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1), "iso": (1, 1, 1)}


@dataclass
class ViewOptions:
    live: bool = False
    video: bool = False
    every: int = 1
    fps: int = 24
    field: str = "speed"
    vectors: str = "none"
    limit: float | None = None
    max_vectors: int = 1000
    geometry: str = "mesh"

    def __post_init__(self):
        if self.every < 1 or self.fps < 1 or self.max_vectors < 1:
            raise ValueError("View interval, FPS and vector budget must be positive")
        if self.limit is not None and (not np.isfinite(self.limit) or self.limit <= 0):
            raise ValueError("Colour maximum must be finite and positive")
        if self.field not in FIELDS or self.vectors not in VECTORS:
            raise ValueError("Unknown rendering field or vector")
        if self.geometry not in {"mesh", "sphere"}:
            raise ValueError("Unknown rendering geometry")

    def preflight(self):
        try:
            import pyvista  # noqa: F401
            if self.video:
                import imageio.v2  # noqa: F401
                import imageio_ffmpeg
                imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError("Install rendering dependencies: pip install -e '.[view]'") from exc


def particle_frame(engine):
    """Restore compensation in FP64 and convert normalized state to SI."""
    state = engine.state()
    def restored(name):
        value = np.asarray(state[name], dtype=float).copy()
        if name + "Remainder" in state:
            value += state[name + "Remainder"]
        return value
    contacts = np.asarray(state["meta"], dtype=np.int64)
    pairs = np.unique(np.sort(contacts[:, :2], axis=1), axis=0)
    coordination = np.bincount(pairs.ravel(), minlength=len(engine.ids))
    return dict(
        position=restored("position")[:, :3] * engine.length_unit,
        rotation=restored("rotation"),
        velocity=restored("velocity")[:, :3] * engine.length_unit / engine.time_unit,
        force=np.asarray(state["force"], float)[:, :3] * engine.force_unit,
        torque=np.asarray(state["torque"], float)[:, :3] * engine.force_unit * engine.length_unit,
        momentum=restored("momentum")[:, :3] * engine.mass_unit * engine.length_unit**2 / engine.time_unit,
        cell=np.asarray(state["cell"], float)[0, :3] * engine.length_unit,
        coordination=coordination,
        shape=engine.ids.copy(),
        step=engine.step,
        time=engine.step * engine.spec.timestep,
    )


def surface_arrays(library, ids, scales, frame):
    """Original grain triangles at their physical pose; no geometry repair."""
    points, faces, owners = [], [], []
    offset = 0
    for i, (idx, scale) in enumerate(zip(ids, scales)):
        shape = library.shapes[int(idx)]
        q = frame["rotation"][i]
        q = q / np.linalg.norm(q)
        vertices = shape.vertices @ quat_matrix(q).T * scale + frame["position"][i]
        points.append(vertices)
        faces.append(np.column_stack((np.full(len(shape.faces), 3), shape.faces + offset)))
        owners.append(np.full(len(shape.faces), i))
        offset += len(vertices)
    return np.concatenate(points), np.concatenate(faces).ravel(), np.concatenate(owners)


class Renderer:
    """One offscreen scene for all movies; a separate interactive live camera."""

    def __init__(self, simulation, output, options):
        import pyvista as pv
        self.pv = pv
        self.sim = simulation
        self.output = Path(output) / "visualization"
        self.output.mkdir(exist_ok=False)
        self.options = options
        self.plotters = []
        self.writers = {}
        self.calls = 0
        self.frames = 0
        self.error = None
        self.last_step = None
        self.live = None
        self.movie = None
        self.initial_cell = None
        try:
            if options.live:
                self.live = pv.Plotter(window_size=(960, 720), title="Sand Atlas DEM")
                self.plotters.append(self.live)
            if options.video:
                import imageio.v2 as imageio
                self.movie = pv.Plotter(off_screen=True, window_size=(960, 720))
                self.plotters.append(self.movie)
                for view in VIEWS:
                    self.writers[view] = imageio.get_writer(
                        str(self.output / f"{view}.mp4"), fps=options.fps,
                        codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                    )
            for plotter in self.plotters:
                plotter.set_background("#18212c")
                plotter.add_axes()
        except Exception:
            self.close()
            raise

    def update(self, *, final=False):
        if self.error is not None:
            return
        try:
            self.calls += 1
            if not final and (self.calls - 1) % self.options.every:
                if self.live is not None and not self.live._closed and self.frames:
                    self.live.update()
                return
            engine = self.sim.engine
            if self.last_step == engine.step:
                return
            frame = particle_frame(engine)
            if self.initial_cell is None:
                self.initial_cell = frame["cell"].copy()
            self._draw(frame)
            self.last_step = engine.step
            self.frames += 1
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"Visualization stopped; simulation continues: {self.error}")

    def _camera(self, plotter, direction):
        centre = self.initial_cell / 2
        direction = np.asarray(direction, float)
        eye = centre + direction / np.linalg.norm(direction) * np.linalg.norm(self.initial_cell) * 2
        up = (0, 1, 0) if direction[2] == 1 and direction[0] == 0 else (0, 0, 1)
        plotter.camera_position = [eye, centre, up]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = float(np.linalg.norm(self.initial_cell) * 0.6)
        plotter.reset_camera_clipping_range()

    def _draw(self, frame):
        pv, options, engine = self.pv, self.options, self.sim.engine
        values = (np.linalg.norm(frame["velocity"], axis=1) if options.field == "speed"
                  else np.linalg.norm(frame[options.field], axis=1)
                  if options.field in {"force", "torque", "momentum"} else frame[options.field])
        maximum = options.limit or float(np.max(values)) or 1.0
        if options.geometry == "mesh":
            points, faces, owners = surface_arrays(self.sim.library, engine.ids, engine.scales, frame)
            mesh = pv.PolyData(points, faces)
            mesh.cell_data["value"] = values[owners]
        else:
            cloud = pv.PolyData(frame["position"])
            cloud["radius"] = np.array([
                self.sim.library.shapes[int(i)].radius * s
                for i, s in zip(engine.ids, engine.scales)
            ])
            cloud["value"] = values
            mesh = cloud.glyph(scale="radius", orient=False, geom=pv.Sphere(radius=1, theta_resolution=12, phi_resolution=8))
        arrows = None
        if options.vectors != "none":
            stride = max(1, int(np.ceil(len(values) / options.max_vectors)))
            vectors = frame[options.vectors][::stride]
            largest = float(np.linalg.norm(vectors, axis=1).max())
            if largest > 0:
                cloud = pv.PolyData(frame["position"][::stride])
                cloud["vector"] = vectors
                arrows = cloud.glyph(orient="vector", scale="vector", factor=float(frame["cell"].min()) * 0.12 / largest)
        cell = frame["cell"]
        box = pv.Box(bounds=(0, cell[0], 0, cell[1], 0, cell[2])).outline()
        for plotter in self.plotters:
            if plotter is self.live and plotter._closed:
                continue
            plotter.add_mesh(mesh, name="grains", scalars="value", cmap="viridis", clim=(0, maximum),
                             scalar_bar_args={"title": FIELDS[options.field], "color": "white"}, reset_camera=False, render=False)
            plotter.add_mesh(box, name="cell", color="white", reset_camera=False, render=False)
            plotter.remove_actor("vectors", render=False)
            if arrows is not None:
                plotter.add_mesh(arrows, name="vectors", color="tomato", reset_camera=False, render=False)
            label = f"Step {frame['step']} | t = {frame['time']:.6g} s | {len(values)} grains"
            label += "\nColour: " + FIELDS[options.field]
            if options.limit is None:
                label += " (auto range per frame)"
            if options.vectors != "none":
                label += f"\n{options.vectors} arrows: relative length, rescaled per frame"
            plotter.add_text(label, name="info", font_size=10, color="white")
        if self.live is not None and not self.live._closed:
            if not self.frames:
                self._camera(self.live, VIEWS["iso"])
                self.live.show(interactive_update=True, auto_close=False)
            self.live.update()
        if self.movie is not None:
            for view, direction in VIEWS.items():
                self._camera(self.movie, direction)
                self.movie.render()
                self.writers[view].append_data(self.movie.screenshot(return_img=True))

    def close(self):
        for resource in [*self.writers.values(), *self.plotters]:
            try:
                resource.close()
            except Exception as exc:
                if self.error is None:
                    self.error = f"{type(exc).__name__}: {exc}"
        report = dict(
            format="sand-dem-visualization/1", status="failed" if self.error else "complete",
            failure=self.error, frames=self.frames, last_step=self.last_step,
            options=vars(self.options),
            views=list(VIEWS) if self.options.video else [],
            timing="One frame per sampled diagnostic boundary; initial and final committed states included. Playback FPS is not physical real time.",
            geometry="Original grain meshes" if self.options.geometry == "mesh" else "Bounding spheres (display proxy only)",
            conventions="SI; peculiar velocity; net particle contact force/torque; periodic cell without ghost copies. No nodal force chains.",
        )
        write_json(self.output / "render.json", report)
        return report
