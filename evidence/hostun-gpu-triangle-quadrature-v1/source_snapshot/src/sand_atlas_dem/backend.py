"""Explicit Metal backend. There is deliberately no CPU fallback."""

import platform
from pathlib import Path

import numpy as np


class Metal:
    def __init__(self, debug=False):
        import slangpy as spy

        self.spy = spy
        self.device = spy.Device(
            type=spy.DeviceType.metal,
            enable_debug_layers=debug,
            compiler_options={
                "include_paths": [str(Path(__file__).parent / "shaders")],
                "floating_point_mode": spy.SlangFloatingPointMode.precise,
            },
        )
        if self.device.info.type != spy.DeviceType.metal:
            raise RuntimeError("Metal device required; fallback is forbidden")
        self.kernels = {}

    def kernel(self, name, module="solver.slang"):
        key = (module, name)
        if key not in self.kernels:
            p = self.device.load_program(
                str(Path(__file__).parent / "shaders" / module), [name]
            )
            self.kernels[key] = self.device.create_compute_kernel(p)
        return self.kernels[key]

    def buffer(self, array=None, *, count=None, stride=4):
        usage = (
            self.spy.BufferUsage.shader_resource | self.spy.BufferUsage.unordered_access
        )
        if array is not None:
            a = np.ascontiguousarray(array)
            return self.device.create_buffer(size=max(4, a.nbytes), usage=usage, data=a)
        return self.device.create_buffer(
            element_count=max(1, count), struct_size=stride, usage=usage
        )

    def dispatch(self, name, n, encoder=None, **kwargs):
        globals_ = {"r": kwargs.pop("r")} if "r" in kwargs else {}
        self.kernel(name).dispatch(
            thread_count=[max(1, int(n)), 1, 1],
            command_encoder=encoder,
            vars=globals_,
            **kwargs,
        )

    def submit(self, encoder):
        return self.device.submit_command_buffer(encoder.finish())

    def read(self, buffer, dtype=np.float32, shape=None):
        a = buffer.to_numpy().view(dtype).reshape(-1).copy()
        return a if shape is None else a.reshape(shape)

    def doctor(self):
        x = np.linspace(-2, 2, 257, dtype=np.float32)
        a = self.buffer(x)
        b = self.buffer(count=len(x))
        self.dispatch("smoke", len(x), N=len(x), a=a, b=b)
        self.device.wait()
        y = self.read(b)
        if not np.allclose(y, x * x + 1, rtol=1e-6, atol=1e-6):
            raise RuntimeError("Metal compute verification failed")
        return dict(
            format="sand-dem-doctor/1",
            backend="metal",
            adapter=str(self.device.info.adapter_name),
            slangpy=self.spy.__version__,
            python=platform.python_version(),
            smoke_elements=len(x),
            smoke_max_error=float(np.max(abs(y - (x * x + 1)))),
            fallback=False,
            floating_point_mode_requested="precise",
        )
