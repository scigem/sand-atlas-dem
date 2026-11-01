"""Experimental immutable Slang root bindings; inputs retained for identity safety."""


class ImmutableBindings:
    def __init__(self, metal):
        self.metal = metal
        self.entries = {}
        self.hits = 0
        self.misses = 0

    def signature(self, value):
        if isinstance(value, dict):
            return tuple((key, self.signature(v)) for key, v in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(self.signature(v) for v in value)
        if isinstance(value, self.metal.spy.Buffer):
            return ("buffer", id(value))
        if value is None or isinstance(value, (str, int, float, bool)):
            return (type(value).__name__, value)
        raise TypeError(f"Unqualified cached binding type: {type(value)}")

    def dispatch(self, kernel, thread_count, command_encoder, vars=None, **kwargs):
        if command_encoder is None:
            return kernel.dispatch(thread_count=thread_count, vars=vars or {}, **kwargs)
        values = vars or {}
        key = (id(kernel), self.signature(values), self.signature(kwargs))
        entry = self.entries.get(key)
        if entry is None:
            root = self.metal.device.create_root_shader_object(kernel.program)
            cursor = self.metal.spy.ShaderCursor(root)
            for name, value in values.items():
                cursor[name].write(value)
            if kwargs:
                parameters = cursor.find_entry_point(0)
                for name, value in kwargs.items():
                    parameters[name].write(value)
            # Retain Python identities as well as immutable native resource bindings.
            entry = (root, kernel, values, kwargs)
            self.entries[key] = entry
            self.misses += 1
        else:
            self.hits += 1
        encoder = command_encoder.begin_compute_pass()
        encoder.bind_pipeline(kernel.pipeline, entry[0])
        encoder.dispatch(thread_count)
        encoder.end()


class BoundKernel:
    def __init__(self, cache, kernel):
        self.cache = cache
        self.kernel = kernel

    def dispatch(self, thread_count, vars=None, command_encoder=None, **kwargs):
        return self.cache.dispatch(
            self.kernel, thread_count, command_encoder, vars, **kwargs
        )
