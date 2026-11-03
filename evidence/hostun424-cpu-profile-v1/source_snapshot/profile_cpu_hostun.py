"""Frozen full-inventory CPU wall-clock profile; never dispatches Metal work."""

import functools
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import h5py

from sand_atlas_dem.cpu import CPU, NativeKernel, PythonKernel
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment

from verify_cpu_hostun import digest, write


def run():
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "studies/hostun424-cpu-profile-v1.json"
    protocol = json.loads(protocol_path.read_text())
    output = root / "runs" / protocol["id"]
    output.mkdir(exist_ok=False)
    snapshot = output / "source_snapshot"
    shutil.copytree(
        root / "src/sand_atlas_dem",
        snapshot / "sand_atlas_dem",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ["profile_cpu_hostun.py", "verify_cpu_hostun.py"]:
        shutil.copy2(root / "examples" / name, snapshot / name)
    shutil.copy2(protocol_path, output / "protocol.json")
    checkpoint = root / protocol["checkpoint"]
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
        str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()
    }
    write(
        output / "freeze.json",
        {"id": protocol["id"], "files": frozen, "private_inputs": inputs},
    )
    report = {
        "id": protocol["id"],
        "backend": "cpu",
        "gpu_dispatches": 0,
        "scientific_admission": False,
        "timing_scope": protocol["instrumentation"],
    }

    def progress(phase):
        write(output / "progress.json", {"phase": phase, "unix_time": time.time()})
        print(phase, flush=True)

    originals = {}
    try:
        progress("loading-private-geometry")
        library = ShapeLibrary.load(library_path)
        with h5py.File(checkpoint) as f:
            state = {k: f[k][:] for k in f}
            state["contacts"] = int(f.attrs["contacts"])
            step, unit = int(f.attrs["step"]), float(f.attrs["length_unit"])
        ids, scales = state.pop("shape_ids"), state.pop("scales")
        engine = Engine(
            library,
            spec,
            state["position"][:, :3] * unit,
            state["cell"][0, :3] * unit,
            ids,
            scales,
            metal=CPU(),
        )
        progress("rebuilding-checkpoint-untimed")
        engine.restore(state, step)
        del state
        before = engine.diagnostics()
        totals = defaultdict(lambda: {"seconds": 0.0, "calls": 0})

        def instrument(original):
            @functools.wraps(original)
            def dispatch(kernel, *, command_encoder=None, **kwargs):
                def timed(operation):
                    def execute():
                        start = time.perf_counter()
                        try:
                            return operation()
                        finally:
                            totals[kernel.name]["seconds"] += (
                                time.perf_counter() - start
                            )
                            totals[kernel.name]["calls"] += 1

                    return execute

                if command_encoder is None:
                    return timed(lambda: original(kernel, **kwargs))()
                start_index = len(command_encoder.operations)
                result = original(kernel, command_encoder=command_encoder, **kwargs)
                for index in range(start_index, len(command_encoder.operations)):
                    command_encoder.operations[index] = timed(
                        command_encoder.operations[index]
                    )
                return result

            return dispatch

        for cls in [NativeKernel, PythonKernel]:
            originals[cls] = cls.dispatch
            cls.dispatch = instrument(cls.dispatch)
        durations = []
        for index in range(protocol["steps"]):
            progress(f"profiling-step-{index + 1}")
            start = time.perf_counter()
            engine.advance(1, rate=protocol["rate_s"], drag=protocol["drag_s"])
            durations.append(time.perf_counter() - start)
            write(
                output / "partial-profile.json",
                {"step_wall_seconds": durations, "kernels": totals},
            )
        for cls, original in originals.items():
            cls.dispatch = original
        total = sum(durations)
        report.update(
            execution_completed=True,
            particles=engine.n,
            unique_shapes=len(library.shapes),
            estimated_solver_bytes=engine.estimated_bytes,
            step_wall_seconds=durations,
            mean_seconds_per_step=total / len(durations),
            kernels={
                name: {
                    **values,
                    "fraction_of_step_wall_time": values["seconds"] / total,
                    "seconds_per_step": values["seconds"] / len(durations),
                }
                for name, values in sorted(
                    totals.items(), key=lambda item: -item[1]["seconds"]
                )
            },
            untimed_kernel_remainder_seconds=total
            - sum(v["seconds"] for v in totals.values()),
            initial_diagnostics=before,
            final_diagnostics=engine.diagnostics(),
        )
        # Preserve a copy of the observed Metal timing windows; no new GPU work.
        telemetry = (
            root / "runs/hostun-relaxed-overlap-equilibration-v5/telemetry.jsonl"
        )
        rows = []
        with telemetry.open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # Ignore a partially appended final line.
        windows = {"early": rows[:101], "recent": rows[-101:]}
        write(output / "observed-metal-windows.json", windows)
        report["observed_metal_comparison"] = {}
        for name, rows in windows.items():
            a, b = rows[0], rows[-1]
            seconds = (b["wall_seconds"] - a["wall_seconds"]) / (b["step"] - a["step"])
            report["observed_metal_comparison"][name] = {
                "start_step": a["step"],
                "end_step": b["step"],
                "seconds_per_step": seconds,
                "cpu_slowdown": total / len(durations) / seconds,
                "active_contacts_at_end": b["active_surface_contacts"],
                "scope": protocol["comparison"],
            }
        progress("complete")
    except Exception as exc:
        report.update(execution_completed=False, failure=repr(exc))
        raise
    finally:
        for cls, original in originals.items():
            cls.dispatch = original
        report["source_integrity"] = all(
            digest(output / p) == h for p, h in frozen.items()
        )
        report["input_integrity"] = all(digest(Path(p)) == h for p, h in inputs.items())
        write(output / "report.json", report)


if __name__ == "__main__":
    run()
