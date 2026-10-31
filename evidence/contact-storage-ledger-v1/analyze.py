"""Execute only frozen allocation declarations with symbolic buffers, never a GPU."""

import ast
from pathlib import Path
from types import SimpleNamespace

from sand_atlas_dem.util import read_json, sha256, write_json


class Buffer:
    def __init__(self, size):
        self.size = size


class Allocator:
    def buffer(self, data=None, *, count=None, stride=4):
        return Buffer(data.nbytes if data is not None else count * stride)


def main():
    out = Path(__file__).resolve().parent
    root = out.parents[1]
    engine_path = root / "src/sand_atlas_dem/gpu.py"
    parent = root / "evidence/hostun-resolution-memory-intake-v1/report.json"
    paths = [engine_path, parent, Path(__file__)]
    frozen = {str(p): sha256(p) for p in paths}
    write_json(out / "prospective-freeze.json", dict(
        input_sha256=frozen, scope=__doc__,
        workload="424, 10000 and 50000 particles; 4096 contacts/particle; original power-of-two hash sizing. No capacities changed.",
        acceptance="Exact declaration-versus-independent-algebra equality. Allocation lower bounds only, not total measured memory or scientific acceptance."))
    tree = ast.parse(engine_path.read_text())
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Engine")
    init = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")

    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
            and t.value.id == "self" and t.attr == name for t in node.targets)

    start = next(i for i, n in enumerate(init.body) if assigns(n, "history_buffers"))
    end = next(i for i, n in enumerate(init.body) if assigns(n, "scan_pairs"))
    backup = next(n for n in init.body if assigns(n, "history_backup"))
    declarations = ast.Module(body=init.body[start:end] + [backup], type_ignores=[])
    code = compile(ast.fix_missing_locations(declarations), str(engine_path), "exec")
    rows = []
    for particles in (424, 10000, 50000):
        capacity = max(16, particles * 4096)
        hash_size = 1 << (max(32, 2 * capacity) - 1).bit_length()
        state = SimpleNamespace(contact_capacity=capacity, hash_size=hash_size,
                                buffers={}, m=Allocator())
        state.allocate = lambda name, count, stride=4: state.buffers.update(
            {name: Buffer(count * stride)})
        symbolic_numpy = SimpleNamespace(
            uint32="uint32", full=lambda count, value, dtype: SimpleNamespace(nbytes=count * 4))
        exec(code, {"self": state, "np": symbolic_numpy})
        components = {
            "two_history_state_sides": sum(b.size for h in state.history_buffers for k, b in h.items() if k != "hash"),
            "history_state_rollback": sum(b.size for k, b in state.history_backup.items() if k != "hash"),
            "three_hash_tables": sum(h["hash"].size for h in state.history_buffers) + state.history_backup["hash"].size,
            "point_force_loss_scratch": sum(b.size for b in state.buffers.values()),
            # Allocated earlier in the constructor: two directed adjacency entries.
            "contact_adjacency": 8 * capacity,
        }
        assert set(state.buffers) == {"pointDepth", "contactForce", "normalLossPower"}
        assert len(state.history_buffers) == 2
        assert set(state.history_backup) == {"meta", "slip", "normalArea", "hash"}
        total = sum(components.values())
        assert total == 188 * capacity + 12 * hash_size
        rows.append(dict(particles=particles, contact_capacity=capacity, hash_slots=hash_size,
                         component_bytes=components, contact_storage_bytes=total,
                         contact_storage_GiB=total / 1024**3,
                         alone_exceeds_36GiB=total > 36 * 1024**3))
    assert all(sha256(p) == h for p, h in frozen.items())
    write_json(out / "report.json", dict(identity=out.name, input_sha256=frozen,
        inputs_unchanged=True, symbolic_allocations_only=True, gpu_executed=False,
        rows=rows, engineering_implication="At 50k, contact storage alone exceeds installed unified memory, independently of geometry resolution. Geometry-only optimization cannot make this configuration fit.",
        limitations=["No real device or host capacity-sized buffer is allocated.",
                     "Excludes geometry, particle/pair/bin state, scans, compiler, driver and host peaks.",
                     "Current capacities are not established as sufficient for a dense specimen.",
                     "No smaller capacity, altered force law, or actual storage optimization is qualified."]))
    print(rows)


if __name__ == "__main__":
    main()
