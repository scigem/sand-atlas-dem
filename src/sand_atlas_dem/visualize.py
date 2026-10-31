"""Offline inspection; simulation has no rendering dependency."""

from pathlib import Path

import h5py
import numpy as np

from .geometry import ShapeLibrary
from .util import quat_matrix, read_json


def inspect_shapes(library, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for i, s in enumerate(library.shapes):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for axis, ax in enumerate(axes):
            section = np.take(s.sdf, s.sdf.shape[axis] // 2, axis=axis)
            ax.imshow(section.T, origin="lower")
            ax.contour(section.T, levels=[0], colors=["white"])
            ax.set_title(f"{s.id}: axis {axis}")
        fig.tight_layout()
        fig.savefig(output / f"shape-{i}.png", dpi=140)
        plt.close(fig)
        import trimesh

        trimesh.Trimesh(s.vertices, s.faces, process=False).export(
            output / f"shape-{i}.ply"
        )


def export_snapshots(run, output):
    import meshio

    run = Path(run)
    output = Path(output)
    manifest = read_json(run / "manifest.json")
    library = ShapeLibrary.load(manifest["experiment"]["shape_library"])
    for file in sorted(run.glob("particles-*.h5")):
        vertices = []
        faces = []
        offset = 0
        with h5py.File(file) as f:
            version = f.attrs.get("format")
            if version not in {"sand-dem-snapshot/1", "sand-dem-snapshot/2"}:
                raise ValueError("Unsupported particle snapshot")
            x = f["position"][:, :3].astype(float)
            q = f["rotation"][:].astype(float)
            if version == "sand-dem-snapshot/2":
                x += f["positionRemainder"][:, :3]
                q += f["rotationRemainder"][:]
            x *= f.attrs["length_unit_m"]
            ids = f["shape_id"][:]
            scale = f["scale"][:]
        for p, rot, idx, s in zip(x, q, ids, scale):
            shape = library.shapes[int(idx)]
            v = shape.vertices @ quat_matrix(rot).T * s + p
            vertices.append(v)
            faces.append(shape.faces + offset)
            offset += len(v)
        meshio.write_points_cells(
            output / (file.stem + ".vtu"),
            np.concatenate(vertices),
            [("triangle", np.concatenate(faces))],
        )
    history = read_json(run / "history.json")["rows"]
    if history:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        strain = [r["axial_strain"] for r in history]
        axes[0].plot(strain, [r["q"] / 1000 for r in history])
        axes[0].set(xlabel="Axial engineering strain", ylabel="q (kPa)")
        axes[1].plot(strain, [r["volumetric_strain"] for r in history])
        axes[1].set(
            xlabel="Axial engineering strain", ylabel="Trace of engineering strains"
        )
        fig.tight_layout()
        fig.savefig(output / "response.png", dpi=160)
        plt.close(fig)
