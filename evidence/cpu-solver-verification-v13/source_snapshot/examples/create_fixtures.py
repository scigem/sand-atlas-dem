"""Generate small original synthetic fixtures; no third-party geometry is bundled."""

import json
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.geometry import ShapeLibrary, from_mesh, sphere


def main():
    out = Path("assets/generated")
    out.mkdir(parents=True, exist_ok=True)
    ShapeLibrary([sphere(resolution=32, samples=512)]).save(out / "sphere.h5")
    polygon = np.array([[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]])
    v = (
        np.vstack(
            [
                np.column_stack([polygon, np.full(6, -0.3)]),
                np.column_stack([polygon, np.full(6, 0.3)]),
            ]
        )
        * 0.001
    )
    triangles = np.array([[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]])
    faces = [*triangles[:, ::-1], *(triangles + 6)]
    for i in range(6):
        j = (i + 1) % 6
        faces.extend([[i, j, j + 6], [i, j + 6, i + 6]])
    mesh = trimesh.Trimesh(v, np.asarray(faces), process=False)
    mesh.export(out / "concave-prism.ply")
    ShapeLibrary(
        [
            from_mesh(
                mesh,
                shape_id="original-concave-prism",
                unit="m",
                resolution=32,
                samples=512,
                provenance={"fixture": "original_extruded_L_polygon"},
            )
        ]
    ).save(out / "concave-prism.h5")
    spec = {
        "format": "sand-dem-experiment/1",
        "shape_library": "../assets/generated/sphere.h5",
        "particles": 64,
        "timestep": 1e-8,
        "initial_solid_fraction": 0.05,
        "diagnostic_interval": 5,
        "snapshot_interval": 10,
        "stages": [{"kind": "hold", "steps": 20, "drag": 0}],
    }
    Path("examples/smoke.json").write_text(json.dumps(spec, indent=2) + "\n")
    mesh_spec = dict(
        spec,
        shape_library="../assets/generated/concave-prism.h5",
        contact_geometry="mesh",
        timestep=1e-9,
        initial_solid_fraction=0.01,
    )
    Path("examples/mesh-smoke.json").write_text(json.dumps(mesh_spec, indent=2) + "\n")
    spec.update(
        particles=10000,
        diagnostic_interval=50,
        snapshot_interval=1000,
        stages=[
            {
                "kind": "isotropic",
                "steps": 2000000,
                "drag": 1e5,
                "pressure": 10000,
                "gain": 1000,
                "filter_time": 1e-5,
                "max_rate": 1000,
                "converged_samples": 100,
            },
            {
                "kind": "triaxial",
                "steps": 20000000,
                "drag": 0,
                "pressure": 10000,
                "gain": 1000,
                "filter_time": 1e-5,
                "max_rate": 1000,
                "rate": 1.0,
                "stop_strain": 0.1,
            },
        ],
    )
    Path("examples/pilot-template.json").write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    main()
