"""Frozen native CPU execution of an unchanged Slang plane-contact kernel."""

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def inputs(count, seed):
    rng = np.random.default_rng(seed)
    points = np.zeros((count, 4), dtype=np.float32)
    points[:, :3] = rng.uniform(-1e-3, 1e-3, (count, 3))
    points[:, 2] = rng.uniform(-5e-5, 1e-5, count)
    points[:, 3] = rng.uniform(1e-9, 1e-7, count)
    velocities = rng.uniform(-0.01, 0.01, (count, 4)).astype(np.float32)
    slips = rng.uniform(-5e-5, 5e-5, (count, 4)).astype(np.float32)
    slips[:, 2] = 0
    slips[:, 3] = rng.integers(0, 2, count)
    return points, velocities, slips


def load_function(path):
    library = ct.CDLL(str(path))
    fn = library.cpu_plane
    fn.argtypes = [ct.c_uint32] + [ct.c_void_p]*6 + [ct.c_float]*4 + [ct.c_uint32]
    fn.restype = None
    return library, fn


def evaluate(fn, arrays, p, repetitions=1):
    points, velocities, slips = arrays
    outputs = [np.zeros_like(points) for _ in range(3)]
    args = [len(points)] + [a.ctypes.data for a in [*arrays, *outputs]]
    args += [p[k] for k in ["kn_Pa_per_m", "kt_Pa_per_m", "mu", "dt_s"]]
    started = time.perf_counter()
    fn(*args, repetitions)
    return outputs, time.perf_counter()-started


def verify(fn, p, reference):
    arrays = inputs(p["verification_contacts"], p["seed"])
    points, velocities, slips = arrays
    actual, _ = evaluate(fn, arrays, p)
    expected = [np.zeros_like(points, dtype=np.float64) for _ in range(3)]
    kn, kt, mu, dt = [float(np.float32(p[k])) for k in ["kn_Pa_per_m", "kt_Pa_per_m", "mu", "dt_s"]]
    for i, (point, velocity, old) in enumerate(zip(points, velocities, slips)):
        if point[2] >= 0:
            continue
        area, depth = float(point[3]), -float(point[2])
        prior = old[:3].astype(float) if old[3] else np.zeros(3)
        force, slip = reference.nodal_force(depth, [0, 0, 1], area, prior,
            velocity[:3].astype(float), 2*kn, 2*kt, mu, dt)
        energy = 0.5*area*(kn*depth**2 + kt*np.dot(slip, slip))
        trial = prior + velocity[:3].astype(float)*[dt, dt, 0]
        ft = force.copy(); ft[2] = 0
        loss = max(0., -np.dot(ft, trial-slip))
        expected[0][i] = [*force, energy]
        expected[1][i] = [*slip, loss]
        expected[2][i] = [*np.cross(point[:3].astype(float), force), 1]
    errors = {}
    for name, a, e, tolerance in zip(["force_energy", "slip_loss", "torque_active"],
                                     actual, expected, p["absolute_tolerances"]):
        np.testing.assert_allclose(a, e, rtol=p["rtol"], atol=tolerance)
        errors[name] = float(np.max(np.abs(a-e)))
    return errors


def worker(library, protocol, output, ready, go):
    p = json.loads(protocol.read_text())
    retained, fn = load_function(library)
    arrays = inputs(p["benchmark_contacts_per_worker"], p["seed"])
    evaluate(fn, arrays, p)  # warm up outside timing
    ready.write_text("ready\n")
    deadline = time.monotonic() + p["timeout_s"]
    while not go.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Benchmark start barrier")
        time.sleep(0.005)
    values, seconds = evaluate(fn, arrays, p, p["repetitions"])
    write(output, {"seconds": seconds, "contacts": p["benchmark_contacts_per_worker"]*p["repetitions"],
                   "checksum": float(np.sum(values[0], dtype=np.float64))})


def run(protocol, output):
    p = json.loads(protocol.read_text())
    root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    sources = {
        "study_cpu_contacts.py": Path(__file__),
        "cpu_plane_bridge.cpp": root / "examples/cpu_plane_bridge.cpp",
        "reference.py": root / "src/sand_atlas_dem/reference.py",
        "planeProbe.cpp": root / p["generated_cpp"],
        "codegen-freeze.json": root / "runs/cpu-slang-feasibility-v1/freeze.json",
    }
    for name, source in sources.items():
        shutil.copy2(source, snapshot / name)
    shutil.copy2(protocol, output / "protocol.json")
    frozen = {str(f.relative_to(output)): digest(f) for f in sorted(output.rglob("*")) if f.is_file()}
    write(output / "freeze.json", {"id": p["id"], "files": frozen,
                                   "created_unix": time.time(), "gpu_dispatches": 0})
    report = {"id": p["id"], "scientific_admission": False,
              "gpu_dispatches": 0, "whole_solver_throughput_measured": False,
              "mesh_query_measured": False}
    try:
        library = output / "plane-contact.dylib"
        command = ["xcrun", "clang++", "-O3", "-std=c++17", "-fno-fast-math",
                   "-dynamiclib", str(snapshot / "cpu_plane_bridge.cpp"), "-o", str(library)]
        compilation = subprocess.run(command, capture_output=True, text=True, timeout=p["timeout_s"])
        (output / "native-compile.log").write_text(compilation.stdout+compilation.stderr)
        report["compile_command"] = command
        report["compile_returncode"] = compilation.returncode
        compilation.check_returncode()
        retained, fn = load_function(library)
        # Relative imports use the package, whose reference hash is bound above.
        from sand_atlas_dem import reference
        report["reference_path"] = reference.__file__
        if digest(reference.__file__) != digest(snapshot / "reference.py"):
            raise RuntimeError("Reference source differs from freeze")
        report["max_absolute_errors"] = verify(fn, p, reference)
        report["contact_verification_passed"] = True
        report["benchmark"] = []
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        for count in p["worker_counts"]:
            folder = output / f"workers-{count}"
            folder.mkdir()
            go = folder / "go"
            processes = []
            logs = []
            try:
                for i in range(count):
                    log = open(folder / f"worker-{i}.log", "w")
                    logs.append(log)
                    processes.append(subprocess.Popen([sys.executable,
                        str(snapshot / "study_cpu_contacts.py"), "--worker", str(library),
                        str(output / "protocol.json"), str(folder / f"worker-{i}.json"),
                        str(folder / f"ready-{i}"), str(go)], stdout=log, stderr=log, env=env))
                deadline = time.monotonic()+p["timeout_s"]
                while not all((folder / f"ready-{i}").exists() for i in range(count)):
                    if time.monotonic() > deadline or any(proc.poll() is not None for proc in processes):
                        raise RuntimeError("CPU worker startup failed")
                    time.sleep(0.01)
                go.write_text("go\n")
                for proc in processes:
                    if proc.wait(timeout=p["timeout_s"]) != 0:
                        raise RuntimeError("CPU worker failed")
                rows = [json.loads((folder / f"worker-{i}.json").read_text()) for i in range(count)]
                # Timing excludes process startup; report each worker's measured kernel time.
                rate = sum(r["contacts"]/r["seconds"] for r in rows)
                report["benchmark"].append({"workers": count, "workers_detail": rows,
                                            "aggregate_kernel_contacts_per_s": rate})
            finally:
                for proc in processes:
                    if proc.poll() is None:
                        proc.kill(); proc.wait()
                for log in logs:
                    log.close()
        report["execution_passed"] = True
    except Exception as exc:
        report["execution_passed"] = False
        report["failure"] = repr(exc)
        raise
    finally:
        report["source_integrity"] = all(digest(output/f) == h for f,h in frozen.items())
        write(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=5, type=Path)
    parser.add_argument("--protocol", type=Path, default=Path("studies/cpu-contact-execution-v1.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
    else:
        if args.output is None:
            parser.error("--output required")
        run(args.protocol.resolve(), args.output.resolve())
