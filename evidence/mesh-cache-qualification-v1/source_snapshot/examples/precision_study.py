"""Compare tiny GPU motions to analytic references, including an uncompensated baseline."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sand_atlas_dem.geometry import ShapeLibrary, sphere
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p = read_json(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    shape = sphere(resolution=32, samples=512)
    spec = Experiment(particles=2, timestep=p["timestep_s"])
    e = Engine(
        ShapeLibrary([shape]),
        spec,
        [[0.001, 0.001, 0.001], [0.004, 0.004, 0.004]],
        [0.006] * 3,
        [0, 0],
        [1, 1],
    )
    q = np.tile(np.array(p["initial_quaternion_xyzw"], np.float32), (2, 1))
    e.buffers["rotation"].copy_from_numpy(q)
    e.initialize()
    velocity = np.tile(
        np.r_[p["initial_normalized_velocity"], 0].astype(np.float32), (2, 1)
    )
    e.buffers["velocity"].copy_from_numpy(velocity)
    w = np.array(p["physical_angular_velocity_rad_s"])
    inertia = shape.inertia * spec.density / (e.mass_unit * e.length_unit**2)
    momentum = inertia @ (w * e.time_unit)
    e.buffers["momentum"].copy_from_numpy(
        np.column_stack([np.tile(momentum, (2, 1)), np.zeros(2)]).astype(np.float32)
    )
    initial = e.state()
    steps = p["steps"]
    duration = steps * spec.timestep
    for start in range(0, steps, 100):
        e.advance(min(100, steps - start), rate=p["diagonal_cell_rate_s"])
    final = e.state()

    def full(name):
        return final[name].astype(float) + final.get(
            name + "Remainder", np.zeros_like(final[name])
        )

    actualx = full("position")[:, 0]
    expectedx = (
        initial["position"][:, 0].astype(float)
        + float(velocity[0, 0]) * duration / e.time_unit
    )
    actualv = full("velocity")[:, 1]
    expectedv = float(velocity[0, 1]) * np.exp(-p["diagonal_cell_rate_s"][1] * duration)
    strain = 1 - float(final["cell"][0, 1]) / float(initial["cell"][0, 1])
    expectedstrain = -np.expm1(p["diagonal_cell_rate_s"][1] * duration)
    expectedq = Rotation.from_rotvec(w * duration) * Rotation.from_quat(q[0])
    actualq = Rotation.from_quat(full("rotation")[0])
    errors = dict(
        translation_absolute_normalized=float(abs(actualx - expectedx).max()),
        velocity_absolute_normalized=float(abs(actualv - expectedv).max()),
        engineering_strain_absolute=float(abs(strain - expectedstrain)),
        orientation_error_rad=float((actualq * expectedq.inv()).magnitude()),
    )
    import sand_atlas_dem

    source = Path(sand_atlas_dem.__file__).parent
    report = dict(
        identity=p["identity"],
        protocol=p,
        protocol_sha256=sha256(args.protocol),
        source_hashes={f.name: sha256(f) for f in source.glob("*.py")},
        shader_hashes={f.name: sha256(f) for f in (source / "shaders").glob("*.slang")},
        backend=e.m.doctor(),
        actual_strain=strain,
        expected_strain=float(expectedstrain),
        errors=errors,
        passed=all(errors[k] <= v for k, v in p["limits"].items()),
    )
    write_json(args.output / "report.json", report)
    print(report["passed"], errors)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
