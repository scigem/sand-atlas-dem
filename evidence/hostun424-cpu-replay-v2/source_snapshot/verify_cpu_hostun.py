"""Explicit full-inventory CPU restart/step study; private geometry stays private."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np

from sand_atlas_dem.cpu import CPU
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(protocol_path, output):
    root = Path(__file__).resolve().parents[1]
    p = json.loads(protocol_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    shutil.copytree(
        root / "src/sand_atlas_dem",
        snapshot / "sand_atlas_dem",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(__file__, snapshot / "verify_cpu_hostun.py")
    shutil.copy2(protocol_path, output / "protocol.json")
    checkpoint = root / p["checkpoint"]
    with h5py.File(checkpoint) as f:
        spec = Experiment(**json.loads(f.attrs["experiment"]))
        expected_library = str(f.attrs["shape_sha256"])
    library_path = Path(spec.shape_library)
    if digest(library_path) != expected_library:
        raise RuntimeError("Shape identity mismatch")
    inputs = {str(checkpoint): digest(checkpoint), str(library_path): expected_library}
    frozen = {
        str(f.relative_to(output)): digest(f) for f in output.rglob("*") if f.is_file()
    }
    write(
        output / "freeze.json",
        dict(id=p["id"], files=frozen, private_inputs=inputs, created_unix=time.time()),
    )
    report = dict(
        id=p["id"], backend="cpu", gpu_dispatches=0, scientific_admission=False
    )

    def progress(phase):
        write(output / "progress.json", dict(phase=phase, unix_time=time.time()))
        print(phase, flush=True)

    try:
        progress("loading-private-library-and-checkpoint")
        library = ShapeLibrary.load(library_path)
        with h5py.File(checkpoint) as f:
            state = {k: f[k][:] for k in f}
            state["contacts"] = int(f.attrs["contacts"])
            step = int(f.attrs["step"])
            unit = float(f.attrs["length_unit"])
        ids = state.pop("shape_ids")
        scales = state.pop("scales")
        progress("allocating-cpu-engine")
        cpu = CPU()
        e = Engine(
            library,
            spec,
            state["position"][:, :3] * unit,
            state["cell"][0, :3] * unit,
            ids,
            scales,
            metal=cpu,
        )
        report.update(
            particles=e.n,
            unique_shapes=len(library.shapes),
            source_nodes=sum(len(s.samples) for s in library.shapes),
            estimated_bytes=e.estimated_bytes,
        )
        progress("rebuilding-frozen-metal-checkpoint-on-cpu")
        start = time.perf_counter()
        e.restore(state, step)
        report["restore_wall_s"] = time.perf_counter() - start
        rebuilt = e.state()
        transfer_passed = True
        for name in ["force", "torque"]:
            old = np.asarray(state[name][:, :3], float)
            new = np.asarray(rebuilt[name][:, :3], float)
            error = float(np.linalg.norm(new - old) / max(np.linalg.norm(old), 1e-30))
            report[name + "_relative_l2_error"] = error
            if error > p[name + "_relative_l2_tolerance"]:
                transfer_passed = False
        report["cross_backend_transfer_passed"] = transfer_passed
        report["old_contacts"] = state["contacts"]
        report["cpu_contacts"] = rebuilt["contacts"]
        if p.get("diagnose_branch_precision"):
            from sand_atlas_dem.cpu import SHADERS, _function_span

            progress("diagnosing-original-fp32-branch-with-same-cpu-scheduling")
            legacy = CPU()
            original = (SHADERS / "solver.slang").read_text()
            a, b = _function_span(legacy.source, "centerBranch")
            ca, cb = _function_span(original, "centerBranch")
            legacy.source = legacy.source[:a] + original[ca:cb] + legacy.source[b:]
            legacy.source_hash = hashlib.sha256(
                (legacy.source_hash + legacy.source).encode()
            ).hexdigest()
            e.m = legacy
            e.restore(state, step)
            alternate = e.state()
            report["original_fp32_branch_diagnostic"] = {}
            for name in ["force", "torque"]:
                old = np.asarray(state[name][:, :3], float)
                new = np.asarray(alternate[name][:, :3], float)
                report["original_fp32_branch_diagnostic"][
                    name + "_relative_l2_error"
                ] = float(np.linalg.norm(new - old) / max(np.linalg.norm(old), 1e-30))
            del alternate
            e.m = cpu
            e.restore(rebuilt, step)
        del state
        progress("advancing-cpu-hold")
        start = time.perf_counter()
        e.advance(p["steps"], rate=p["rate_s"], drag=p["drag_s"])
        report["advance_wall_s"] = time.perf_counter() - start
        expected = e.state()
        report["final_diagnostics"] = e.diagnostics()
        progress("restoring-and-replaying-cpu-hold")
        e.restore(rebuilt, step)
        e.advance(p["steps"], rate=p["rate_s"], drag=p["drag_s"])
        repeated = e.state()
        errors = {}
        for name in [
            "position",
            "velocity",
            "rotation",
            "momentum",
            "force",
            "torque",
            "accumulated",
            "slip",
            "normalArea",
        ]:
            np.testing.assert_allclose(
                repeated[name],
                expected[name],
                rtol=p["cpu_replay_rtol"],
                atol=p["cpu_replay_atol"],
            )
            errors[name] = float(np.max(np.abs(repeated[name] - expected[name])))
        np.testing.assert_array_equal(repeated["meta"], expected["meta"])
        report.update(
            passed=transfer_passed,
            cpu_execution_and_replay_passed=True,
            replay_max_absolute_errors=errors,
            backend_identity=cpu.doctor(),
        )
        progress("complete")
    except Exception as exc:
        report.update(passed=False, failure=repr(exc))
        raise
    finally:
        report["source_integrity"] = all(
            digest(output / f) == h for f, h in frozen.items()
        )
        report["input_integrity"] = all(digest(Path(f)) == h for f, h in inputs.items())
        write(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("studies/hostun424-cpu-replay-v1.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    run(a.protocol.resolve(), a.output.resolve())
