"""Explicit, isolated Slang CPU probes. Never selects a GPU or advances dynamics."""

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


def cpp_probe(shader, entry, output):
    """Use the bundled compiler's compatibility C API for an isolated codegen probe."""
    import slangpy

    package = Path(slangpy.__file__).parent
    libpath = next(package.glob("libslang-compiler*.dylib"))
    lib = ct.CDLL(str(libpath))
    signatures = {
        "spCreateSession": (ct.c_void_p, [ct.c_char_p]),
        "spDestroySession": (None, [ct.c_void_p]),
        "spCreateCompileRequest": (ct.c_void_p, [ct.c_void_p]),
        "spDestroyCompileRequest": (None, [ct.c_void_p]),
        "spProcessCommandLineArguments": (
            ct.c_int, [ct.c_void_p, ct.POINTER(ct.c_char_p), ct.c_int]
        ),
        "spCompile": (ct.c_int, [ct.c_void_p]),
        "spGetDiagnosticOutput": (ct.c_char_p, [ct.c_void_p]),
        "spGetEntryPointCode": (
            ct.c_void_p, [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_size_t)]
        ),
    }
    for name, (result, args) in signatures.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = result, args
    session = lib.spCreateSession(None)
    request = lib.spCreateCompileRequest(session)
    try:
        args = [str(shader), "-entry", entry, "-stage", "compute", "-target", "cpp"]
        encoded = (ct.c_char_p * len(args))(*(a.encode() for a in args))
        parsed = lib.spProcessCommandLineArguments(request, encoded, len(args))
        status = lib.spCompile(request) if parsed >= 0 else parsed
        diagnostics = (lib.spGetDiagnosticOutput(request) or b"").decode(errors="replace")
        print(diagnostics, flush=True)
        code = b""
        if status >= 0:
            size = ct.c_size_t()
            pointer = lib.spGetEntryPointCode(request, 0, ct.byref(size))
            if pointer:
                code = ct.string_at(pointer, size.value)
                Path(output).write_bytes(code)
        print(json.dumps({"status": status, "code_bytes": len(code),
                          "compiler": str(libpath), "compiler_sha256": digest(libpath)}))
        return 0 if status >= 0 and code else 1
    finally:
        lib.spDestroyCompileRequest(request)
        lib.spDestroySession(session)


def device_probe(shaders, protocol):
    import slangpy as spy

    device = spy.Device(type=spy.DeviceType.cpu,
                        compiler_options={"include_paths": [str(shaders)]})
    if device.info.type != spy.DeviceType.cpu:
        raise RuntimeError("Explicit CPU device required")
    print("CPU device:", device.info, flush=True)
    program = device.load_program(str(shaders / "solver.slang"), ["smoke"])
    kernel = device.create_compute_kernel(program)
    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    x = np.linspace(-2, 2, protocol["smoke_elements"], dtype=np.float32)
    a = device.create_buffer(size=x.nbytes, usage=usage, data=x)
    b = device.create_buffer(size=x.nbytes, usage=usage)
    kernel.dispatch(thread_count=[len(x), 1, 1], N=len(x), a=a, b=b)
    device.wait()
    y = b.to_numpy().view(np.float32).reshape(-1)
    np.testing.assert_allclose(y, x*x+1, rtol=protocol["smoke_rtol"],
                               atol=protocol["smoke_atol"])
    print(json.dumps({"smoke_passed": True}))


def run(protocol_path, output):
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(protocol_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    shutil.copytree(root / "src/sand_atlas_dem/shaders", snapshot / "shaders")
    shutil.copy2(__file__, snapshot / "study_cpu_slang.py")
    shutil.copy2(root / "src/sand_atlas_dem/reference.py", snapshot / "reference.py")
    shutil.copy2(protocol_path, output / "protocol.json")
    write(output / "freeze.json", {
        "id": protocol["id"], "created_unix": time.time(),
        "files": {str(p.relative_to(output)): digest(p)
                  for p in sorted(output.rglob("*")) if p.is_file()},
        "cpu_only": True, "gpu_dispatches": 0,
    })
    runner = snapshot / "study_cpu_slang.py"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    results = []

    def probe(name, args):
        started = time.perf_counter()
        try:
            result = subprocess.run([sys.executable, str(runner), *args],
                                    env=env, capture_output=True, text=True,
                                    timeout=protocol["timeout_s"])
            log = result.stdout + result.stderr
            record = dict(name=name, returncode=result.returncode)
        except subprocess.TimeoutExpired as exc:
            def decode(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
            log = decode(exc.stdout) + decode(exc.stderr)
            record = dict(name=name, returncode=None, timeout=True)
        (output / f"{name}.log").write_text(log)
        record["seconds"] = time.perf_counter() - started
        results.append(record)
        write(output / "report.json", dict(id=protocol["id"], probes=results,
              cpu_only=True, gpu_dispatches=0, scientific_admission=False,
              whole_solver_throughput_measured=False))
        print(json.dumps(record), flush=True)

    probe("cpu-device", ["--device-probe", str(snapshot / "shaders"),
                         "--protocol", str(output / "protocol.json")])
    for module, entry in protocol["probes"]:
        probe(f"cpp-{entry}", ["--cpp-probe", str(snapshot / "shaders" / module),
                               entry, str(output / f"{entry}.cpp")])
    freeze = json.loads((output / "freeze.json").read_text())
    intact = all(digest(output / p) == h for p, h in freeze["files"].items())
    report = json.loads((output / "report.json").read_text())
    report["source_integrity"] = intact
    write(output / "report.json", report)
    if not intact:
        raise RuntimeError("Frozen source changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path,
                        default=Path("studies/cpu-slang-feasibility-v1.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device-probe", type=Path)
    parser.add_argument("--cpp-probe", nargs=3)
    args = parser.parse_args()
    if args.cpp_probe:
        sys.exit(cpp_probe(*args.cpp_probe))
    elif args.device_probe:
        device_probe(args.device_probe, json.loads(args.protocol.read_text()))
    else:
        if args.output is None:
            parser.error("--output required")
        run(args.protocol.resolve(), args.output.resolve())
