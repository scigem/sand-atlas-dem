"""Independent analytical box gradient probe of frozen shader candidates, Metal only."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.gpu import Engine, f4
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    modules = {
        name: args.study / "source_snapshot/sand_atlas_dem/shaders" / (name + ".slang")
        for name in ["baseline", "precision"]
    }
    protocol = dict(
        identity=out.name,
        distance_atol=1e-6,
        normal_atol=2e-5,
        depths=[1e-2, 1e-4, 1e-6, 1e-7],
        modules={str(p): sha256(p) for p in modules.values()},
        driver_sha256=sha256(__file__),
        scope="Analytical signed box gradients from actual FP32 query coordinates; interior-face, exterior-face, edge and corner points. Same scalar distance path. Diagnostic comparison, no scientific admission.",
    )
    write_json(out / "protocol.json", protocol)
    (out / "driver.py").write_bytes(Path(__file__).read_bytes())
    shape = from_mesh(
        trimesh.creation.box(extents=[100] * 3), unit="um", resolution=8, samples=32
    )
    m = Metal()
    e = Engine(
        ShapeLibrary([shape]),
        Experiment(particles=1, timestep=1e-9, contact_geometry="mesh"),
        [[0.005] * 3],
        [0.01] * 3,
        [0],
        [1],
        metal=m,
    )
    half = np.float32(shape.vertices.max(0) / e.length_unit).astype(float)
    points = []
    for axis in range(3):
        for sign in [-1, 1]:
            for depth in protocol["depths"]:
                for side in [-1, 1]:
                    point = half * np.array([0.17, -0.31, 0.23])
                    point[axis] = sign * (half[axis] + side * depth)
                    points.append(point)
    points.extend(
        [
            half * np.array(x)
            for x in [
                [1.2, 1.3, 0.2],
                [-1.2, 1.3, 0.2],
                [1.2, 1.3, 1.4],
                [-1.2, -1.3, -1.4],
            ]
        ]
    )
    points = np.array(points, np.float32).astype(float)
    q = abs(points) - half
    positive = np.maximum(q, 0)
    distance = np.linalg.norm(positive, axis=1) + np.minimum(q.max(axis=1), 0)
    normals = np.zeros_like(points)
    for i in range(len(points)):
        if np.any(q[i] > 0):
            normals[i] = positive[i] * np.sign(points[i]) / np.linalg.norm(positive[i])
        else:
            axis = np.argmax(q[i])
            normals[i, axis] = np.sign(points[i, axis])
    a = m.buffer(f4(points))
    b = m.buffer(count=len(points), stride=16)
    records = {}
    for name, module in modules.items():
        m.kernel("meshProbe", str(module.resolve())).dispatch(
            thread_count=[len(points), 1, 1],
            r=e.bindings(),
            count=len(points),
            shape=0,
            points=a,
            result=b,
        )
        m.device.wait()
        actual = m.read(b).reshape(-1, 4)
        de = float(np.max(abs(actual[:, 3] - distance)))
        ne = float(np.max(abs(actual[:, :3] - normals)))
        records[name] = dict(
            distance_max_abs=de,
            normal_max_abs=ne,
            passed=de <= protocol["distance_atol"] and ne <= protocol["normal_atol"],
        )
        np.savez(
            out / (name + ".npz"),
            points=points,
            expected_distance=distance,
            expected_normal=normals,
            actual=actual,
        )
    write_json(
        out / "report.json",
        dict(
            results=records,
            backend=m.doctor(),
            sources_unchanged=all(
                sha256(p) == h for p, h in protocol["modules"].items()
            ),
            scientific_admission=False,
        ),
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
