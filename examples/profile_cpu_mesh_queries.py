"""Instrument a private generated native collector; production source stays intact."""

import ctypes as ct
import hashlib
import json
import re
import subprocess
from pathlib import Path


NAMES = [
    "nodeContact",
    "meshDistance",
    "meshMayContain",
    "meshClusterMayContact",
    "meshBoxDistance",
    "closestTriangle",
    "nearest_updates",
    "popped_nodes",
]


def install(kernel, output, *, counts_only=False):
    if kernel.backend.threads != 1:
        raise ValueError("Native query instrumentation requires exactly one CPU thread")
    folder = (
        kernel.backend.cache
        / hashlib.sha256(
            (kernel.backend.source_hash + kernel.name).encode()
        ).hexdigest()
    )
    original = (folder / "kernel.cpp").read_text()
    declarations = """#include <chrono>
#include <cstdint>
#include <cstring>
static thread_local uint64_t profile_calls[8] = {};
static thread_local uint64_t profile_ns[8] = {};
struct QueryScope {
    int index;
    std::chrono::steady_clock::time_point start;
    QueryScope(int i):index(i),start(std::chrono::steady_clock::now()) { ++profile_calls[i]; }
    ~QueryScope() { profile_ns[index] += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count(); }
};
"""
    cpp = original
    for index, name in enumerate(NAMES[:6]):
        pattern = r"(^static [^\n]*\b" + name + r"_\d+\([^\n]*\)\s*\{)"
        code = (
            f"QueryScope query_profile({index});"
            if index < 4 and not counts_only
            else f"++profile_calls[{index}];"
        )
        cpp, count = re.subn(
            pattern, lambda m: m[0] + "\n" + code, cpp, flags=re.MULTILINE
        )
        if count != 1:
            raise RuntimeError(f"Expected one generated {name} definition, got {count}")
    sites = (
        []
        if counts_only
        else [
            (6, r"^.*Vector<float, 3>[^\n]*meshNormals_\d+\.Load[^\n]*$"),
            (7, r"^.*float [^\n]*meshBoxDistance_\d+\(pending_[^\n]*$"),
        ]
    )
    for index, pattern in sites:
        cpp, count = re.subn(
            pattern,
            lambda m: f"++profile_calls[{index}];\n" + m[0],
            cpp,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError(
                f"Expected one generated profile site {index}, got {count}"
            )
    cpp = (
        declarations
        + cpp
        + """
extern "C" void cpu_profile_reset() { std::memset(profile_calls,0,sizeof(profile_calls)); std::memset(profile_ns,0,sizeof(profile_ns)); }
extern "C" void cpu_profile_read(uint64_t* calls,uint64_t* nanos) { std::memcpy(calls,profile_calls,sizeof(profile_calls)); std::memcpy(nanos,profile_ns,sizeof(profile_ns)); }
"""
    )
    path = output / "instrumented-collector.cpp"
    path.write_text(cpp)
    binary = output / "instrumented-collector.dylib"
    command = kernel.backend.compiler + [
        "-std=c++17",
        "-O3",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-shared",
        "-fPIC",
        "-pthread",
        str(path),
        "-o",
        str(binary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (output / "instrumented-compile.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Instrumented native collector compilation failed")
    library = ct.CDLL(str(binary))
    function = library.cpu_dispatch
    function.argtypes, function.restype = kernel.fn.argtypes, kernel.fn.restype
    kernel.fn = function
    reset = library.cpu_profile_reset
    reset.argtypes, reset.restype = [], None
    read = library.cpu_profile_read
    read.argtypes, read.restype = [ct.POINTER(ct.c_uint64)] * 2, None
    identity = dict(
        original_cpp_sha256=hashlib.sha256(original.encode()).hexdigest(),
        instrumented_cpp_sha256=hashlib.sha256(cpp.encode()).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        command=command,
        timing="inclusive native function timing; overlapping scopes must not be summed; box/triangle/update/pop counters only; instrumentation overhead included",
        counts_only=counts_only,
    )
    if counts_only:
        identity["timing"] = (
            "Call counters only for six native functions; no timers/update/pop instrumentation. Performance admission uses the separate uninstrumented study."
        )
    (output / "instrumented-identity.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )

    def counters():
        calls, nanos = (ct.c_uint64 * 8)(), (ct.c_uint64 * 8)()
        read(calls, nanos)
        return {
            name: {
                "calls": int(calls[i]),
                "inclusive_seconds": float(nanos[i]) / 1e9
                if i < 4 and not counts_only
                else None,
            }
            for i, name in enumerate(NAMES[:6] if counts_only else NAMES)
        }

    return reset, counters, identity
