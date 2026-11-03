"""Frozen GPU verification of immutable bindings under mutable caller inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from immutable_bindings import ImmutableBindings

from sand_atlas_dem.backend import Metal
from sand_atlas_dem.util import sha256, write_json

SHADER = r"""
struct Config {uint N;float factor;};
struct Resources {Config c;StructuredBuffer<float> input;RWStructuredBuffer<float> output;};
ParameterBlock<Resources> r;
[shader("compute")][numthreads(64,1,1)]
void apply(uint id:SV_DispatchThreadID,uniform float offset){
    if(id<r.c.N)r.output[id]=r.c.factor*r.input[id]+offset;
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    module = (out / "probe.slang").resolve()
    module.write_text(SHADER)
    helper = Path(__file__).with_name("immutable_bindings.py")
    p = dict(
        identity=out.name,
        shader_sha256=sha256(module),
        driver_sha256=sha256(__file__),
        helper_sha256=sha256(helper),
        gate="Exact FP32 output vs independent integer-valued analytical results; caller dictionaries mutated between queued dispatches, globals and entrypoint parameters change, buffers change identity/content, and zero-length work leaves output untouched. Cache hit required for repeated full binding; new entry required for scalar or resource identity changes. No numerical tolerance or physics change.",
    )
    write_json(out / "protocol.json", p)
    (out / "driver.py").write_bytes(Path(__file__).read_bytes())
    (out / "immutable_bindings.py").write_bytes(helper.read_bytes())
    m = Metal()
    kernel = m.kernel("apply", str(module))
    cache = ImmutableBindings(m)
    source = m.buffer(np.array([1, 2, 3, 4], np.float32))
    a = m.buffer(np.full(4, -99, np.float32))
    b = m.buffer(np.full(4, -99, np.float32))
    config = {"N": 4, "factor": 2.0}
    resources = {"c": config, "input": source, "output": a}
    records = []
    encoder = m.device.create_command_encoder()
    cache.dispatch(kernel, [4, 1, 1], encoder, {"r": resources}, offset=1.0)
    config["factor"] = 3.0
    resources["output"] = b
    cache.dispatch(kernel, [4, 1, 1], encoder, {"r": resources}, offset=-2.0)
    config.update(N=2, factor=4.0)
    resources["output"] = a
    cache.dispatch(kernel, [4, 1, 1], encoder, {"r": resources}, offset=5.0)
    m.submit(encoder)
    m.device.wait()
    assert np.array_equal(m.read(a), [9, 13, 7, 9])
    assert np.array_equal(m.read(b), [1, 4, 7, 10])
    assert cache.misses == 3
    records.append(dict(case="queued_mutated_config_and_outputs", passed=True))
    # Buffer contents change while object identity and complete binding remain cacheable.
    source.copy_from_numpy(np.array([2, 3, 4, 5], np.float32))
    config.update(N=4, factor=2.0)
    before = cache.hits
    encoder = m.device.create_command_encoder()
    cache.dispatch(kernel, [4, 1, 1], encoder, {"r": resources}, offset=1.0)
    m.submit(encoder)
    m.device.wait()
    assert cache.hits == before + 1
    assert np.array_equal(m.read(a), [5, 7, 9, 11])
    records.append(dict(case="content_change_reuses_binding", passed=True))
    replacement = m.buffer(np.array([10, 20, 30, 40], np.float32))
    resources.update(input=replacement, output=b)
    config["factor"] = 3.0
    before = cache.misses
    encoder = m.device.create_command_encoder()
    cache.dispatch(kernel, [4, 1, 1], encoder, {"r": resources}, offset=-2.0)
    m.submit(encoder)
    m.device.wait()
    assert cache.misses == before + 1
    assert np.array_equal(m.read(b), [28, 58, 88, 118])
    records.append(dict(case="resource_identity_invalidates", passed=True))
    # Revert to an earlier complete binding; its original root must be immutable.
    resources.update(input=source, output=a)
    config.update(N=4, factor=2.0)
    before = cache.hits
    encoder = m.device.create_command_encoder()
    cache.dispatch(kernel, [4, 1, 1], encoder, {"r": resources}, offset=1.0)
    m.submit(encoder)
    m.device.wait()
    assert cache.hits == before + 1
    assert np.array_equal(m.read(a), [5, 7, 9, 11])
    records.append(dict(case="revert_reuses_original_root", passed=True))
    config["N"] = 0
    encoder = m.device.create_command_encoder()
    cache.dispatch(kernel, [4, 1, 1], encoder, {"r": resources}, offset=100.0)
    m.submit(encoder)
    m.device.wait()
    assert np.array_equal(m.read(a), [5, 7, 9, 11])
    records.append(dict(case="zero_length_scalar_change", passed=True))
    # Original caller dict has changed repeatedly; cached lifetime roots retain snapshots.
    first = next(iter(cache.entries.values()))
    assert first[2]["r"]["input"] is source and first[2]["r"]["c"] == {
        "N": 4,
        "factor": 2.0,
    }
    records.append(dict(case="retained_binding_tree_is_snapshot", passed=True))
    write_json(
        out / "report.json",
        dict(
            passed=True,
            checks=records,
            hits=cache.hits,
            misses=cache.misses,
            backend=m.doctor(),
            sources_unchanged=sha256(module) == p["shader_sha256"]
            and sha256(out / "driver.py") == p["driver_sha256"]
            and sha256(out / "immutable_bindings.py") == p["helper_sha256"],
            scientific_admission=False,
        ),
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
