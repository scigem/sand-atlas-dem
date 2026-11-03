"""Compare frozen CPU mesh collectors on identical private Hostun state."""

import importlib.util
import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
from verify_cpu_hostun import digest, write

from sand_atlas_dem.cpu import CPU
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment


def run():
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "studies/hostun424-cpu-cached-bounds-v1.json"
    p = json.loads(protocol_path.read_text())
    output = root / "runs" / p["id"]
    output.mkdir(exist_ok=False)
    snapshot = output / "source_snapshot"
    shutil.copytree(
        root / "src/sand_atlas_dem",
        snapshot / "sand_atlas_dem",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ["compare_cpu_collectors.py", "verify_cpu_hostun.py"]:
        shutil.copy2(root / "examples" / name, snapshot / name)
    baseline_path = root / p["baseline_package"]
    shutil.copytree(
        baseline_path,
        snapshot / "baseline",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    # Freeze a self-contained baseline, and reject unrelated equation changes.
    baseline_path = snapshot / "baseline"
    for path in (snapshot / "sand_atlas_dem").rglob("*"):
        if path.is_file() and path.name != "cpu_scheduling.slang":
            relative = path.relative_to(snapshot / "sand_atlas_dem")
            if digest(path) != digest(baseline_path / relative):
                raise RuntimeError(
                    f"Unexpected baseline equation/runtime difference: {relative}"
                )
    shutil.copy2(protocol_path, output / "protocol.json")
    checkpoint = root / p["checkpoint"]
    with h5py.File(checkpoint) as f:
        spec = Experiment(**json.loads(f.attrs["experiment"]))
        expected_library = str(f.attrs["shape_sha256"])
    library_path = Path(spec.shape_library)
    inputs = {
        str(checkpoint): digest(checkpoint),
        str(library_path): digest(library_path),
    }
    if inputs[str(library_path)] != expected_library:
        raise RuntimeError("Shape identity mismatch")
    frozen = {
        str(f.relative_to(output)): digest(f) for f in output.rglob("*") if f.is_file()
    }
    write(
        output / "freeze.json",
        {"id": p["id"], "files": frozen, "private_inputs": inputs},
    )
    report = {
        "id": p["id"],
        "backend": "cpu",
        "gpu_dispatches": 0,
        "scientific_admission": False,
        "variants": {},
    }

    def progress(phase):
        write(output / "progress.json", {"phase": phase, "unix_time": time.time()})
        print(phase, flush=True)

    def compare(reference, actual):
        checks = {
            name: bool(np.array_equal(value, actual[name]))
            for name, value in reference.items()
            if isinstance(value, np.ndarray)
        }
        checks["contacts"] = reference["contacts"] == actual["contacts"]
        return {"all_bitwise_equal": all(checks.values()), "fields": checks}

    try:
        progress("loading-private-geometry")
        library = ShapeLibrary.load(library_path)
        with h5py.File(checkpoint) as f:
            state = {k: f[k][:] for k in f}
            state["contacts"] = int(f.attrs["contacts"])
            step, unit = int(f.attrs["step"]), float(f.attrs["length_unit"])
        ids, scales = state.pop("shape_ids"), state.pop("scales")
        adapter = CPU()
        engine = Engine(
            library,
            spec,
            state["position"][:, :3] * unit,
            state["cell"][0, :3] * unit,
            ids,
            scales,
            metal=adapter,
        )
        module_spec = importlib.util.spec_from_file_location(
            "frozen_cpu", baseline_path / "cpu.py"
        )
        baseline_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(baseline_module)
        baseline = baseline_module.CPU()
        progress("compiling-collectors-untimed")
        kernels = {
            "baseline": baseline.kernel("collectMeshFrontiers"),
            "cached_bounds": adapter.kernel("collectMeshFrontiers"),
        }
        report["source_identities"] = {
            "baseline": baseline.source_hash,
            "cached_bounds": adapter.source_hash,
        }
        key = ("solver.slang", "collectMeshFrontiers")
        for name, kernel in kernels.items():
            adapter.kernels[key] = kernel
            progress(f"{name}-restoring-checkpoint-untimed")
            start = time.perf_counter()
            engine.restore(state, step)
            restore_seconds = time.perf_counter() - start
            initial = engine.state()
            if name == "baseline":
                reference_initial = initial
            else:
                report["initial_state_comparison"] = compare(reference_initial, initial)
                if not report["initial_state_comparison"]["all_bitwise_equal"]:
                    raise RuntimeError("Initial-state equality gate failed")
            del initial
            durations, collector_seconds = [], []
            original_dispatch = kernel.dispatch

            def dispatch(
                *,
                command_encoder=None,
                _original=original_dispatch,
                _timings=collector_seconds,
                **kwargs,
            ):
                if command_encoder is None:
                    raise RuntimeError("Profiling requires queued execution")
                first = len(command_encoder.operations)
                result = _original(command_encoder=command_encoder, **kwargs)
                for index in range(first, len(command_encoder.operations)):
                    operation = command_encoder.operations[index]

                    def timed(operation=operation):
                        start = time.perf_counter()
                        try:
                            return operation()
                        finally:
                            _timings.append(time.perf_counter() - start)

                    command_encoder.operations[index] = timed
                return result

            kernel.dispatch = dispatch
            try:
                for index in range(p["steps_per_variant"]):
                    progress(f"{name}-step-{index + 1}")
                    start = time.perf_counter()
                    engine.advance(1, rate=p["rate_s"], drag=p["drag_s"])
                    durations.append(time.perf_counter() - start)
            finally:
                kernel.dispatch = original_dispatch
            final = engine.state()
            report["variants"][name] = {
                "step_wall_seconds": durations,
                "mean_seconds_per_step": float(np.mean(durations)),
                "collector_wall_seconds": collector_seconds,
                "checkpoint_reconstruction_seconds_excluded": restore_seconds,
                "final_diagnostics": engine.diagnostics(),
            }
            if name == "baseline":
                reference_final = final
            else:
                report["final_state_comparison"] = compare(reference_final, final)
                if not report["final_state_comparison"]["all_bitwise_equal"]:
                    raise RuntimeError("Final-state equality gate failed")
            write(output / "report.json", report)
        report["step_speedup"] = (
            report["variants"]["baseline"]["mean_seconds_per_step"]
            / report["variants"]["cached_bounds"]["mean_seconds_per_step"]
        )
        report["passed"] = True
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
    run()
