"""Native Atlas replay of the existing analytical bridge fixture; no DEM validation."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from sand_atlas_dem.util import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = read_json(args.protocol)
    root = Path(__file__).resolve().parents[1]
    atlas = Path(policy["atlas_root"])
    handoff = Path(policy["handoff"])
    request_path = handoff / "replay-request.json"
    if sha256(request_path) != policy["request_sha256"]:
        raise ValueError("Frozen replay request changed")
    receipt = read_json(handoff / "handoff.json")
    if receipt["status"] != "complete" or receipt["kind"] != "synthetic-dem":
        raise ValueError("Complete synthetic handoff required")
    inputs = {str(args.protocol.resolve()): sha256(args.protocol)}
    for name, digest in receipt["artifacts"].items():
        if Path(name).name != name or sha256(handoff / name) != digest:
            raise ValueError("Handoff artifact differs from receipt")
        inputs[str((handoff / name).resolve())] = digest
    inputs[str((handoff / "handoff.json").resolve())] = sha256(handoff / "handoff.json")
    solver = Path(policy["solver"]).resolve()
    if sha256(solver) != policy["solver_sha256"]:
        raise ValueError("Frozen native solver changed")
    inputs[str(solver)] = policy["solver_sha256"]
    sources = [
        *sorted((atlas / "atlas").glob("*.py")),
        *sorted((atlas / "schemas").glob("*.json")),
        atlas / "models/linear-elastic.json",
    ]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = {}
    for path in [Path(__file__), args.protocol.resolve()]:
        name = "dem/" + str(path.relative_to(root))
        frozen[name] = sha256(path)
        dest = output / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        inputs[str(path.resolve())] = sha256(path)
    for path in sources:
        name = "atlas/" + str(path.relative_to(atlas))
        frozen[name] = sha256(path)
        dest = output / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        inputs[str(path.resolve())] = sha256(path)
    shutil.copy2(request_path, output / "replay-request.json")
    shutil.copy2(handoff / "virtual-test.json", output / "virtual-test.json")
    write_json(
        output / "prospective-freeze.json",
        dict(inputs=inputs, sources=frozen, policy=policy, scope=__doc__),
    )
    command = [
        sys.executable,
        "-m",
        "atlas.cli",
        "replay",
        str(request_path.resolve()),
        "--solver",
        str(solver),
        "--output",
        str(output / "native-response.json"),
    ]
    with (output / "native.log").open("w") as log:
        run = subprocess.run(
            command,
            cwd=atlas,
            env={**os.environ, "PYTHONPATH": str(atlas)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    response = read_json(output / "native-response.json")
    request = read_json(request_path)
    virtual = read_json(handoff / "virtual-test.json")
    if request["model"] != "linear-elastic":
        raise ValueError("Analytical study requires its declared linear-elastic model")
    young, nu = request["parameters"][1:3]
    initial_stress = np.array(request["initial"][:6])
    initial_strain = np.array(request["path"]["initial_strain"])
    initial_void = request["initial"][6]
    points = response.get("points", [])
    rows = []
    for index, (knot, point, observed) in enumerate(
        zip(request["path"]["knots"], points, virtual["rows"])
    ):
        axial, radial_y, radial_z = knot["targets"][:3]
        delta_y, delta_z = radial_y - initial_stress[1], radial_z - initial_stress[2]
        delta_x = young * (axial - initial_strain[0]) + nu * (delta_y + delta_z)
        stress = initial_stress + [delta_x, delta_y, delta_z, 0, 0, 0]
        strain = initial_strain + [
            axial - initial_strain[0],
            (delta_y - nu * (delta_x + delta_z)) / young,
            (delta_z - nu * (delta_x + delta_y)) / young,
            0,
            0,
            0,
        ]
        void = (1 + initial_void) * np.exp(-sum(strain[:3] - initial_strain[:3])) - 1
        actual_stress = np.array(point["state"][:6])
        actual_strain = np.array(point["strain"])
        q = actual_stress[0] - (actual_stress[1] + actual_stress[2]) / 2
        rows.append(
            dict(
                source_row_index=index,
                axial_strain=observed["axial_strain"],
                synthetic_q_Pa=observed["q"],
                continuum_q_Pa=float(q),
                synthetic_volumetric_strain=observed["volumetric_strain"],
                continuum_volumetric_strain=float(sum(actual_strain[:3])),
                maximum_stress_error_Pa=float(max(abs(actual_stress - stress))),
                maximum_strain_error=float(max(abs(actual_strain - strain))),
                void_ratio_error=abs(float(point["state"][6] - void)),
            )
        )
    complete = len(points) == len(request["path"]["knots"]) == len(virtual["rows"])
    unchanged = all(sha256(path) == digest for path, digest in inputs.items())
    errors = {
        key: max((r[key] for r in rows), default=float("inf"))
        for key in [
            "maximum_stress_error_Pa",
            "maximum_strain_error",
            "void_ratio_error",
        ]
    }
    passed = bool(
        run.returncode == 0
        and response.get("ok") is True
        and complete
        and unchanged
        and all(np.isfinite(list(errors.values())))
        and all(errors[key] <= policy[key] for key in errors)
    )
    write_json(
        output / "report.json",
        dict(
            identity=policy["identity"],
            passed=passed,
            complete=complete,
            source_unchanged=unchanged,
            rows=rows,
            errors=errors,
            scientific_admission=False,
            kind="synthetic-dem-software-fixture",
            note="Actual native continuum execution, independently checked using Hooke compliance and exponential void evolution. Input fixture is analytical, not a GPU-produced or qualified DEM specimen. No fitting or experimental validation.",
        ),
    )
    print("passed:", passed, "points:", len(points), "errors:", errors)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
