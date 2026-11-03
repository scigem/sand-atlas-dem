"""Frozen CPU qualification and independent-process full-step smoke workload."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(protocol_path, output):
    root = Path(__file__).resolve().parents[1]
    p = json.loads(protocol_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in ["src", "tests"]:
        shutil.copytree(
            root / name,
            snapshot / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copy2(__file__, snapshot / "verify_cpu_solver.py")
    # Existing synthetic concave fixture used by an independent geometry check.
    fixture = Path("assets/generated/concave-prism.h5")
    (snapshot / fixture.parent).mkdir(parents=True)
    shutil.copy2(root / fixture, snapshot / fixture)
    shutil.copy2(protocol_path, output / "protocol.json")
    files = {
        str(f.relative_to(output)): digest(f)
        for f in sorted(output.rglob("*"))
        if f.is_file()
    }
    write(
        output / "freeze.json",
        {"id": p["id"], "files": files, "created_unix": time.time()},
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(snapshot / "src"),
        SAND_DEM_METAL="0",
        SAND_DEM_MESH_CACHE="/private/tmp/sand-dem-systematic-cache",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    report = dict(
        id=p["id"],
        backend="cpu",
        gpu_dispatches=0,
        scientific_admission=False,
        whole_solver_workload="synthetic isolated mesh pairs; no dense Hostun performance claim",
    )
    try:
        command = [sys.executable, "-m", "pytest", "tests", *p["pytest_arguments"]]
        with (output / "tests.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=snapshot,
                env=env,
                stdout=log,
                stderr=log,
                timeout=p["timeouts_s"]["tests"],
            )
        report["tests_returncode"] = result.returncode
        write(output / "report.json", report)
        if result.returncode:
            raise RuntimeError("Frozen CPU/regression tests failed; see tests.log")
        # Run fixture generation under the frozen source too.
        generation = """from sand_atlas_dem.geometry import ShapeLibrary,from_mesh,with_face_quadrature
import trimesh
shape=from_mesh(trimesh.creation.box(extents=[.001,.0008,.0006]),unit="m",resolution=12,samples=64,provenance={"source":"synthetic CPU full-step workload","coordinate_units":"m","repairs":[]})
shape=with_face_quadrature(shape,rule="gauss3",subdivisions=2)
ShapeLibrary([shape]).save(r"LIBRARY")
""".replace("LIBRARY", str(output / "fixture.h5"))
        result = subprocess.run(
            [sys.executable, "-c", generation],
            cwd=snapshot,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        (output / "fixture-generation.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        report["fixture_sha256"] = digest(output / "fixture.h5")
        benchmark = p["benchmark"]
        groups = []
        for count in benchmark["worker_counts"]:
            group = output / f"workers-{count}"
            group.mkdir()
            processes = []
            logs = []
            started = time.perf_counter()
            try:
                for i in range(count):
                    log = (group / f"job-{i}.log").open("w")
                    logs.append(log)
                    command = [
                        sys.executable,
                        "-m",
                        "sand_atlas_dem.cli",
                        "benchmark",
                        "--backend",
                        "cpu",
                        "--output",
                        str(group / f"job-{i}"),
                        "--library",
                        str(output / "fixture.h5"),
                        "--particles",
                        str(benchmark["particles"]),
                        "--steps",
                        str(benchmark["steps"]),
                        "--repeats",
                        str(benchmark["repeats"]),
                        "--timestep",
                        str(benchmark["timestep_s"]),
                        "--packing",
                        benchmark["packing"],
                        "--contact-geometry",
                        benchmark["geometry"],
                    ]
                    processes.append(
                        subprocess.Popen(
                            command, cwd=snapshot, env=env, stdout=log, stderr=log
                        )
                    )
                for process in processes:
                    if process.wait(timeout=p["timeouts_s"]["job"]):
                        raise RuntimeError("CPU benchmark job failed; see job logs")
                elapsed = time.perf_counter() - started
                jobs = [
                    json.loads((group / f"job-{i}/benchmark.json").read_text())
                    for i in range(count)
                ]
                if any(j["backend"]["backend"] != "cpu" for j in jobs):
                    raise RuntimeError("Incorrect backend")
                if any(
                    j["final_diagnostics"]["active_grain_pairs"]
                    != benchmark["particles"] // 2
                    for j in jobs
                ):
                    raise RuntimeError("Frozen workload contact topology changed")
                groups.append(
                    dict(
                        workers=count,
                        whole_group_elapsed_s=elapsed,
                        median_step_seconds=[
                            j["median_seconds_per_step"] for j in jobs
                        ],
                        aggregate_kernel_steps_per_s=sum(
                            j["steps_per_second"] for j in jobs
                        ),
                        completed_jobs=count,
                        job_steps=benchmark["steps"] * benchmark["repeats"],
                    )
                )
                report["groups"] = groups
                write(output / "report.json", report)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                for log in logs:
                    log.close()
        report["passed"] = True
    except Exception as exc:
        report.update(passed=False, failure=repr(exc))
        raise
    finally:
        report["source_integrity"] = all(
            digest(output / f) == h for f, h in files.items()
        )
        write(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("studies/cpu-solver-verification-v1.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())
