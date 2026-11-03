"""Explicit native CPU backend for the same Slang DEM equations.

A case executes serial native kernels. Independent processes provide job parallelism.
There is no implicit selection, GPU device creation, or CPU fallback.
"""

import ctypes as ct
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile

import numpy as np


SHADERS = Path(__file__).parent / "shaders"


def _function_span(source, name):
    match = re.search(r"\b(?:void|float3)\s+" + re.escape(name) + r"\s*\(", source)
    if match is None:
        raise RuntimeError(f"CPU source contract missing {name}")
    start = match.start()
    # Include the compute attributes attached to this function.
    prefix = source[:start]
    attributes = re.search(
        r'(?:\[shader\("compute"\)\]\s*\[numthreads\([^]]*\)\]\s*)$', prefix
    )
    if attributes:
        start = attributes.start()
    opening = source.index("{", match.end())
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return start, end


def cpu_source():
    source = (SHADERS / "solver.slang").read_text()
    scheduling = (SHADERS / "cpu_scheduling.slang").read_text()
    atomic = scheduling[: scheduling.index("// Preserve the local branch")]
    a, b = _function_span(source, "centerBranch")
    ca, cb = _function_span(scheduling, "centerBranch")
    source = source[:a] + scheduling[ca:cb] + source[b:]
    a, b = _function_span(scheduling, "cpuCollectMesh")
    helper = scheduling[a:b]
    for name in [
        "collectMeshContacts",
        "collectMeshFrontiers",
        "collectMeshSubtree",
        "countContacts",
        "emitContacts",
        "collectWallContacts",
    ]:
        a, b = _function_span(source, name)
        replacement = (
            ""
            if name == "collectMeshSubtree"
            else scheduling.split(f"// BEGIN {name}\n")[1].split(f"// END {name}")[0]
        )
        if name == "collectMeshContacts":
            replacement = helper + "\n" + replacement
        source = source[:a] + replacement + source[b:]
    for old, new in [
        ("InterlockedAdd", "cpuAtomicAdd"),
        ("InterlockedOr", "cpuAtomicOr"),
        ("InterlockedExchange", "cpuAtomicExchange"),
        ("InterlockedCompareExchange", "cpuAtomicCompareExchange"),
    ]:
        source = source.replace(old, new)
    return atomic + "\n" + source


def _compile_cpp(shader, entry, output):
    import slangpy

    package = Path(slangpy.__file__).parent
    candidates = sorted(package.glob("libslang-compiler*.dylib")) + sorted(
        package.glob("libslang-compiler*.so*")
    )
    if not candidates:
        raise RuntimeError("Bundled Slang compiler library not found")
    library = candidates[0]
    lib = ct.CDLL(str(library))
    signatures = {
        "spCreateSession": (ct.c_void_p, [ct.c_char_p]),
        "spDestroySession": (None, [ct.c_void_p]),
        "spCreateCompileRequest": (ct.c_void_p, [ct.c_void_p]),
        "spDestroyCompileRequest": (None, [ct.c_void_p]),
        "spProcessCommandLineArguments": (
            ct.c_int,
            [ct.c_void_p, ct.POINTER(ct.c_char_p), ct.c_int],
        ),
        "spCompile": (ct.c_int, [ct.c_void_p]),
        "spSetTargetFloatingPointMode": (None, [ct.c_void_p, ct.c_int, ct.c_uint]),
        "spGetDiagnosticOutput": (ct.c_char_p, [ct.c_void_p]),
        "spGetEntryPointCode": (
            ct.c_void_p,
            [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_size_t)],
        ),
    }
    for name, (result, args) in signatures.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = result, args
    session = lib.spCreateSession(None)
    if not session:
        raise RuntimeError("Slang CPU compiler session creation failed")
    request = lib.spCreateCompileRequest(session)
    try:
        args = [
            str(shader),
            "-entry",
            entry,
            "-stage",
            "compute",
            "-target",
            "cpp",
            "-I",
            str(SHADERS),
        ]
        encoded = (ct.c_char_p * len(args))(*(a.encode() for a in args))
        status = lib.spProcessCommandLineArguments(request, encoded, len(args))
        if status >= 0:
            # Same explicit Slang request as Metal; do not rely on target defaults.
            lib.spSetTargetFloatingPointMode(request, 0, 2)  # PRECISE, pinned slang.h
            status = lib.spCompile(request)
        diagnostics = (lib.spGetDiagnosticOutput(request) or b"").decode(
            errors="replace"
        )
        output.with_suffix(".log").write_text(diagnostics)
        if status < 0:
            raise RuntimeError(f"CPU Slang compilation failed: {entry}\n{diagnostics}")
        size = ct.c_size_t()
        ptr = lib.spGetEntryPointCode(request, 0, ct.byref(size))
        if not ptr or not size.value:
            raise RuntimeError("Slang returned no CPU code")
        output.write_bytes(ct.string_at(ptr, size.value))
    finally:
        lib.spDestroyCompileRequest(request)
        lib.spDestroySession(session)


def _struct(cpp, stem):
    match = re.search(r"struct (" + stem + r"_\d+)\s*\{(.*?)\n\};", cpp, re.S)
    if match is None:
        raise RuntimeError(f"Generated CPU ABI missing {stem}")
    fields = {}
    # Generated #line annotations occur between fields on some compiler versions.
    clean = re.sub(r"^#.*$", "", match[2], flags=re.M)
    for declaration in clean.split(";"):
        field = re.search(r"(.+?)\s+(\w+_\d+)\s*$", declaration.strip(), re.S)
        if field:
            fields[re.sub(r"_\d+$", "", field[2])] = (field[2], field[1].strip())
    return match[1], fields


def _wrapper(cpp, name, width):
    """Bind generated C++ structures by compiler-emitted member identity.

    We do not guess the alignment or the ABI of vector/resource parameter blocks.
    The native compiler constructs them; Python passes only flat primitive arrays.
    """

    def optional(stem):
        if re.search(r"struct " + stem + r"_\d+\s*\{", cpp) is None:
            return None, {}
        return _struct(cpp, stem)

    _, configs = optional("Config")
    resources_type, resources = optional("Resources")
    global_type, globals_ = optional("GlobalParams")
    params_type, params = optional("EntryPointParams")
    if globals_ and set(globals_) != {"r"}:
        raise RuntimeError("Unexpected CPU globals ABI")
    names = [n for n in resources if n != "c"]
    values, uniform_values, uniform_buffers = [], [], []
    code = []
    if resources_type:
        code.append(f"{resources_type} res={{}};")
    if global_type:
        code.append(f"{global_type} globals={{}};globals.{globals_['r'][0]}=&res;")
    if params_type:
        code.append(f"{params_type} params={{}};")

    def scalar_fields(fields, prefix, output, offset=0):
        for n, (field, typ) in fields.items():
            if "StructuredBuffer" in typ:
                continue
            vector = re.search(r"Vector<[^,]+,\s*(\d+)\s*>", typ)
            axes = range(int(vector[1])) if vector else [None]
            for axis in axes:
                output.append([n, axis])
                suffix = "" if axis is None else f"[{axis}]"
                code.append(
                    f"{prefix}.{field}{suffix}=values[{offset + len(output) - 1}];"
                )

    if configs:
        scalar_fields(configs, "res." + resources["c"][0], values)
    scalar_fields(params, "params", uniform_values, len(values))
    for n, (_, typ) in params.items():
        if "StructuredBuffer" in typ:
            uniform_buffers.append(n)
    for i, (prefix, field) in enumerate(
        [("res", resources[n][0]) for n in names]
        + [("params", params[n][0]) for n in uniform_buffers]
    ):
        code.append(
            f"{prefix}.{field}.data=reinterpret_cast<decltype({prefix}.{field}.data)>(buffers[{i}]);"
        )
        code.append(
            f"{prefix}.{field}.count=bytes[{i}]/sizeof(*{prefix}.{field}.data);"
        )
    code.append(
        f"ComputeVaryingInput vi={{}};vi.endGroupID={{(count+{width - 1})/{width},1,1}};"
    )
    ep = "&params" if params_type else "nullptr"
    gp = "&globals" if global_type else "nullptr"
    code.append(f"{name}(&vi,{ep},{gp});")
    wrapper = (
        '\nextern "C" void cpu_dispatch(uint32_t count, void** buffers, uint64_t* bytes, double* values){\n'
        + "\n".join(code)
        + "\n}\n"
    )
    return wrapper, dict(
        buffers=names,
        values=values,
        uniform_buffers=uniform_buffers,
        uniform_values=uniform_values,
    )


class NativeKernel:
    def __init__(self, backend, name):
        self.backend, self.name = backend, name
        key = hashlib.sha256((backend.source_hash + name).encode()).hexdigest()
        folder = backend.cache / key
        folder.mkdir(parents=True, exist_ok=True)
        binary = folder / (
            "kernel.dylib" if platform.system() == "Darwin" else "kernel.so"
        )
        with (folder / "build.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not binary.exists():
                source = folder / "solver.slang"
                source.write_text(backend.source)
                cpp_path = folder / "kernel.cpp"
                _compile_cpp(source, name, cpp_path)
                cpp = cpp_path.read_text()
                wrapper, metadata = _wrapper(cpp, name, backend.thread_width(name))
                cpp_path.write_text(cpp + wrapper)
                temporary = folder / ("building" + binary.suffix)
                command = backend.compiler + [
                    "-std=c++17",
                    "-O3",
                    "-fno-fast-math",
                    "-ffp-contract=off",
                    "-shared",
                    "-fPIC",
                    str(cpp_path),
                    "-o",
                    str(temporary),
                ]
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=120
                )
                (folder / "native-compile.log").write_text(
                    result.stdout + result.stderr
                )
                if result.returncode:
                    raise RuntimeError(
                        f"CPU native compilation failed for {name}: {result.stderr}"
                    )
                metadata.update(
                    key=key,
                    cpp_sha256=hashlib.sha256(cpp_path.read_bytes()).hexdigest(),
                    binary_sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(),
                    command=command,
                )
                (folder / "abi.json").write_text(json.dumps(metadata))
                temporary.replace(binary)
        self.abi = json.loads((folder / "abi.json").read_text())
        self.library = ct.CDLL(str(binary))
        self.fn = self.library.cpu_dispatch
        self.fn.argtypes = [
            ct.c_uint32,
            ct.POINTER(ct.c_void_p),
            ct.POINTER(ct.c_uint64),
            ct.POINTER(ct.c_double),
        ]
        self.fn.restype = None

    def dispatch(self, *, thread_count, command_encoder=None, vars=None, **kwargs):
        bindings = (vars or {}).get("r", kwargs.pop("r", None))
        if self.abi["buffers"] and bindings is None:
            raise ValueError("CPU solver dispatch requires resources")

        def execute():
            count = int(thread_count[0])
            if self.name in {"countContacts", "emitContacts", "collectMeshContacts"}:
                count //= 64
            elif self.name == "collectMeshFrontiers":
                count = min(
                    int(bindings["control"].array.view(np.uint32)[1]),
                    bindings["c"]["pairCapacity"],
                )
            elif self.name == "collectWallContacts":
                count //= 64
            arrays = [bindings[n].array for n in self.abi["buffers"]]
            arrays.extend(kwargs[n].array for n in self.abi["uniform_buffers"])
            pointers = (ct.c_void_p * len(arrays))(*(a.ctypes.data for a in arrays))
            sizes = (ct.c_uint64 * len(arrays))(*(a.nbytes for a in arrays))
            c = bindings["c"] if bindings is not None else {}
            scalars = [
                c[n] if axis is None else c[n][axis] for n, axis in self.abi["values"]
            ]
            scalars.extend(
                kwargs[n] if axis is None else kwargs[n][axis]
                for n, axis in self.abi["uniform_values"]
            )
            values = (ct.c_double * len(scalars))(*scalars)
            self.fn(count, pointers, sizes, values)

        if command_encoder is None:
            execute()
        else:
            command_encoder.operations.append(execute)


class Buffer:
    def __init__(self, array=None, *, count=None, stride=4):
        self.array = (
            np.ascontiguousarray(array).copy().view(np.uint8).reshape(-1)
            if array is not None
            else np.zeros(max(4, count * stride), np.uint8)
        )
        self.size = self.array.nbytes

    def copy_from_numpy(self, array):
        data = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        if data.size != self.size:
            raise ValueError("CPU buffer upload size mismatch")
        self.array[:] = data

    def to_numpy(self):
        return self.array.copy()


class Encoder:
    def __init__(self):
        self.operations = []

    def copy_buffer(self, dest, dest_offset, source, source_offset, size):
        if (
            min(dest_offset, source_offset, size) < 0
            or dest_offset + size > dest.size
            or source_offset + size > source.size
        ):
            raise ValueError("CPU buffer copy out of bounds")
        self.operations.append(
            lambda: np.copyto(
                dest.array[dest_offset : dest_offset + size],
                source.array[source_offset : source_offset + size],
            )
        )


class PythonKernel:
    def __init__(self, name):
        self.name = name

    def dispatch(self, *, thread_count, command_encoder=None, **kwargs):
        def execute():
            n = kwargs["N"]
            if self.name == "blockScan":
                a = kwargs["input"].array.view(np.uint32)[:n]
                out = kwargs["output"].array.view(np.uint32)
                totals = kwargs["totals"].array.view(np.uint32)
                for group, start in enumerate(range(0, n, 256)):
                    chunk = a[start : start + 256].astype(np.uint64)
                    cumulative = np.cumsum(chunk)
                    if cumulative[-1] > np.iinfo(np.uint32).max:
                        raise OverflowError("CPU prefix sum overflow")
                    out[start : start + len(chunk)] = cumulative - chunk
                    totals[group] = cumulative[-1]
            elif self.name == "addBlocks":
                out = kwargs["output"].array.view(np.uint32)
                offsets = kwargs["offsets"].array.view(np.uint32)
                out[:n] += offsets[np.arange(n) // 256]
            else:
                rows = {"reduceRows": 8, "reduceFabric": 2, "reduceWallRows": 3}[
                    self.name
                ]
                walls = thread_count[1] if self.name == "reduceWallRows" else 1
                a = kwargs["input"].array.view(np.float32).reshape(walls, n, rows, 4)
                groups = (n + 255) // 256
                out = (
                    kwargs["output"]
                    .array.view(np.float32)
                    .reshape(walls, groups, rows, 4)
                )
                # Match the GPU binary reduction tree, including its maximum/count fields.
                for wall in range(walls):
                    for group, start in enumerate(range(0, n, 256)):
                        values = np.zeros((256, rows, 4), np.float32)
                        values[: min(256, n - start)] = a[wall, start : start + 256]
                        counts = (
                            values[:, 0, 3].copy().view(np.uint32)
                            if self.name == "reduceWallRows"
                            else None
                        )
                        if counts is not None:
                            values[:, 0, 3] = 0
                        step = 128
                        while step:
                            maximum = (
                                np.maximum(
                                    values[:step, 4, 3], values[step : 2 * step, 4, 3]
                                )
                                if self.name == "reduceRows"
                                else None
                            )
                            values[:step] += values[step : 2 * step]
                            if maximum is not None:
                                values[:step, 4, 3] = maximum
                            if counts is not None:
                                counts[:step] += counts[step : 2 * step]
                            step //= 2
                        if counts is not None:
                            values[0, 0, 3] = counts[0].view(np.float32)
                        out[wall, group] = values[0]

        if command_encoder is None:
            execute()
        else:
            command_encoder.operations.append(execute)


class CPU:
    """Metal-shaped host adapter backed exclusively by native CPU kernels."""

    name = "cpu"
    buffer_type = Buffer

    def __init__(self, *, cache=None):
        self.source = cpu_source()
        import slangpy

        self.compiler = (
            ["xcrun", "clang++"]
            if platform.system() == "Darwin"
            else [os.environ.get("CXX", "c++")]
        )
        compiler_version = subprocess.run(
            self.compiler + ["--version"], capture_output=True, text=True, check=True
        ).stdout
        package = Path(slangpy.__file__).parent
        libraries = sorted(package.glob("libslang-compiler*.dylib")) + sorted(
            package.glob("libslang-compiler*.so*")
        )
        if not libraries:
            raise RuntimeError("Bundled Slang CPU compiler not found")
        self.compiler_sha256 = hashlib.sha256(libraries[0].read_bytes()).hexdigest()
        self.source_hash = hashlib.sha256(
            (
                self.source
                + Path(__file__).read_text()
                + slangpy.__version__
                + self.compiler_sha256
                + compiler_version
                + platform.machine()
                + "".join(p.read_text() for p in sorted(SHADERS.glob("*.slang")))
            ).encode()
        ).hexdigest()
        self.cache = Path(
            cache
            or os.environ.get(
                "SAND_DEM_CPU_CACHE", Path(tempfile.gettempdir()) / "sand-dem-cpu-cache"
            )
        )
        self.cache.mkdir(parents=True, exist_ok=True)
        self.kernels = {}
        self.device = self

    def thread_width(self, name):
        source = (
            self.source
            + (SHADERS / "normal.slang").read_text()
            + (SHADERS / "plane.slang").read_text()
        )
        match = re.search(
            r"\[numthreads\((\d+),\s*1,\s*1\)\]\s*void\s+" + re.escape(name) + r"\b",
            source,
        )
        if not match:
            raise RuntimeError(f"CPU kernel width missing: {name}")
        return int(match[1])

    def kernel(self, name, module="solver.slang"):
        if module not in {
            "solver.slang",
            "normal.slang",
            "plane.slang",
            "scan.slang",
            "diagnostic.slang",
        }:
            raise ValueError(f"Unsupported CPU shader module: {module}")
        key = module, name
        if key not in self.kernels:
            self.kernels[key] = (
                PythonKernel(name)
                if module in {"scan.slang", "diagnostic.slang"}
                else NativeKernel(self, name)
            )
        return self.kernels[key]

    def buffer(self, array=None, *, count=None, stride=4):
        return Buffer(array, count=count, stride=stride)

    def dispatch(self, name, n, encoder=None, **kwargs):
        self.kernel(name).dispatch(
            thread_count=[max(1, int(n)), 1, 1], command_encoder=encoder, **kwargs
        )

    def create_command_encoder(self):
        return Encoder()

    def submit(self, encoder):
        for operation in encoder.operations:
            operation()
        encoder.operations.clear()

    def wait(self):
        pass

    def read(self, buffer, dtype=np.float32, shape=None):
        array = buffer.array.view(dtype).copy()
        return array if shape is None else array.reshape(shape)

    def read_prefix(self, buffer, count, dtype=np.float32, shape=None):
        if (
            type(count) is not int
            or count < 0
            or count * np.dtype(dtype).itemsize > buffer.size
        ):
            raise ValueError("CPU buffer prefix out of bounds")
        array = buffer.array[: count * np.dtype(dtype).itemsize].view(dtype).copy()
        return array if shape is None else array.reshape(shape)

    def write_prefix(self, buffer, array):
        data = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        if data.size > buffer.size:
            raise ValueError("CPU buffer prefix out of bounds")
        buffer.array[: data.size] = data

    def doctor(self):
        x = np.linspace(-2, 2, 257, dtype=np.float32)
        a, b = self.buffer(x), self.buffer(np.zeros_like(x))
        self.dispatch("smoke", len(x), N=len(x), a=a, b=b)
        result = self.read(b)
        if not np.allclose(result, x * x + 1, rtol=1e-6, atol=1e-6):
            raise RuntimeError("CPU compiled kernel smoke check failed")
        return dict(
            format="sand-dem-doctor/1",
            backend="cpu",
            fallback=False,
            adapter=platform.processor() or platform.machine(),
            numerics="native Slang C++ FP32, compensated state, serial kernels per process",
            source_sha256=self.source_hash,
            compiler_sha256=self.compiler_sha256,
            floating_point_mode_requested="precise",
            smoke_elements=257,
            smoke_passed=True,
            smoke_max_error=float(np.max(np.abs(result - (x * x + 1)))),
            job_parallelism="independent processes",
        )
