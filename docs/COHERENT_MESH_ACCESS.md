# Coherent exact-mesh access experiment

The first OpenVDB-inspired query-coherence experiment gives **1.021×** throughput
on the actual dilute Hostun checkpoint: **8.181 → 8.013 ms/step** on Apple M4 Max.
All 13 simulation comparison checks and eight independent query-probe cases pass.
This is below the prospectively fixed 1.05× screening gate and far below the 2×
target. The candidate is not promoted; production shader and dispatch stay unchanged.

## What was implemented

`examples/coherent_mesh_shader.py` generates an experimental shader from the exact
production source. Each lane retains its last winning target triangle while walking
a source subtree. At the next query, it evaluates that triangle at the new point
and uses its squared distance, padded upward by `max(1e-12, abs(d)*1e-5)`, as an
initial search bound. The full original nearest-first BVH traversal still runs,
with the original candidate ordering and strict closer-than predicate. No old
closest point, normal, penetration, contact key or force is reused. Seeds reset
on each target/source subtree; no extra persistent GPU buffer is required.

This is a closest-triangle hint inspired by spatially coherent accessors. It is
not a NanoVDB reader, a cached tree-branch implementation, a sparse SDF or a change
to contact geometry. The bound padding is an engineering margin qualified on the
bounded tests below, not a general formal FP32 error proof.

Both benchmark modes use the same earlier experimental immutable binding helper.
Thus the measured baseline includes the host-overhead optimization, avoiding an
artificial gain from comparing different dispatch paths. Original linked-list force
gather remains enabled in both modes. No source sample ordering or friction history
identity changes.

## Frozen protocol and results

`studies/hostun-coherent-mesh-v1.json` retains the original checkpoint/library hashes,
2 ns checkpoint timestep, loading, drag, 16 warmups, three forward/reverse repetitions
and 128 timed steps per sample. Six restored samples per mode are timed. Numerical
limits are unchanged: position/velocity/rotation 1e-5, normalized force/torque 1e-4,
history rtol 1e-4/atol 1e-8, normals rtol 1e-6/atol 1e-7 and exact contact keys.
All package sources, generated shaders, helpers, protocol and private inputs were
hashed before execution. The report verifies they remain unchanged.

| Measurement | Baseline | Coherent hint |
|---|---:|---:|
| Whole-step median | 8.181 ms | 8.013 ms |
| Synchronized contact-stage median | 7.282 ms | 7.032 ms |
| Exact distance queries, diagnostic step | 7,601 | 7,601 |
| BVH node visits | 381,597 | 315,651 |
| Triangles tested inside BVH | 220,555 | 218,148 |
| Additional seed-triangle evaluations | 0 | 7,292 |

Node visits fall 17.28%, but BVH triangle work falls only 1.09%. Including the seed
itself, triangle evaluations increase from 220,555 to 225,440 (2.21%). This helps
explain the small throughput change. The diagnostic counters are collected in a
separate instrumented step and are excluded from timings. Synchronized stage
measurements include encoding/submission/wait overhead; they are **not GPU timestamps**.
Split-stage execution is separately compared with ordinary batched execution in
both modes and passes the unchanged numerical gates.

The first sandboxed invocation could not create a Metal adapter. Its `run.log` is
preserved. The same untouched frozen study then ran with GPU access outside the
sandbox; `metal-run.log` retains that execution. No CPU simulation was substituted.

## Independent bounded query checks

`examples/probe_coherent_mesh.py` freezes its source, generated shaders and private
concave asset before GPU execution. Two packed shapes (box and concave prism) each
use 1,024 random points, alternating cold and arbitrary valid triangle seeds, then
four rounds that jump between regions while retaining the seed. Shape offsets are
exercised. All 8,192 queried results match the original GPU distance/normal vector
bitwise. Independent FP64 analytical box distances and trimesh triangle distances
agree within 6.99e-11 m, against a frozen 2e-9 m bound.

This checks query software and distances, not contact quadrature convergence,
material behavior or scientific admission. Long trajectories, explicit geometric
degeneracies, walls, damping and production cache integration remain outside scope.
The existing production regression suite was not rerun because production code
was unchanged.

## Interpretation and next experiment

A last-triangle bound is too weak to justify promotion. It trims traversal without
materially reducing triangle evaluation. The stronger VDB-related direction is a
body-frame brick index storing **conservative triangle candidate lists**, shared
across instances of the same shape. Exact triangle tests must still resolve distance
and pseudonormal sign; oversized or uncertified bricks must fall back to the BVH.
One possible construction uses a valid center distance upper bound U and brick
radius R: the nearest triangle for any point in that brick must lie within U+2R
of the center. Candidate construction must account conservatively for FP32 bounds
and preserve deterministic tie behavior. A narrow-band sparse SDF remains a separate
approximate-geometry study because the earlier shallow-overlap SDF accuracy gate
failed. Neither direction has been measured here.

Relevant upstream designs:
[NanoVDB static topology and layout](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/doc/nanovdb/FAQ.md),
[cached accessors](https://www.openvdb.org/documentation/doxygen/ValueAccessor_8h.html),
[portable HLSL/GLSL reader](https://github.com/AcademySoftwareFoundation/openvdb/blob/master/nanovdb/nanovdb/PNanoVDB.h).

Reproduce serially on Metal with a new output identity:

```sh
PYTHONPATH=src python examples/study_coherent_mesh.py \
  studies/hostun-coherent-mesh-v1.json --output evidence/NEW-STUDY
PYTHONPATH=src python examples/probe_coherent_mesh.py \
  --study evidence/NEW-STUDY --output evidence/NEW-PROBE
```

Exact archives/reports: `evidence/hostun-coherent-mesh-v1/` and
`evidence/coherent-mesh-query-probe-v1/`. Local work only; no commit or push.
