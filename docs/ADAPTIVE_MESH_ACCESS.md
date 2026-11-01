# Adaptive triangle-tree experiment

**No qualifying speedup.** Three separately frozen adaptive variants complete all
39 simulation comparisons successfully, but each still sends all 7,601 Hostun
diagnostic queries to the original BVH. Fast-path correctness is exercised by
independent synthetic probes, not established on the Hostun specimen. The final
surface-only variant increases median step time **8.189 → 9.068 ms**. No variant
meets the fixed 5% screen or 2× target; production remains unchanged.

This experiment replaces the rejected uniform brick directory with an immutable,
body-frame octree shared by instances of each shape. Original triangle closest-point
algebra, feature pseudonormals, contact samples/keys, timestep and force law are
retained. It adds no SDF interpolation, geometry repair or NanoVDB dependency.
Production code and defaults remain unchanged.

## Frozen design and bounds

`examples/adaptive_mesh.py` starts from a shape-enclosing box and subdivides an
overfull cell into eight children. A usable leaf contains at most **24 triangles**.
The maximum depth is **8** (nominal resolution 256 per axis) and the fixed allocation
budget is **100,000 nodes per shape**. Deep interior cells, exhausted budgets and
overfull cells at maximum depth use the original complete BVH. No list is truncated.
The builder uses deterministic depth-first allocation; it can spend a shape's
budget unevenly across regions. This limitation is recorded rather than silently
changing allocation after seeing performance.

As in the uniform-brick experiment, a sphere of radius R surrounds each cell and a
nearest mesh vertex supplies upper bound U on center-to-surface distance. A nearest
triangle at any point inside the sphere must have center distance at most U+2R.
Triangle AABB filtering therefore retains every potential winner. Each child filters
its parent's candidates using its own global-vertex upper bound. At runtime **every
ancestor sphere** is checked before a leaf list is trusted. This maintains the
inherited completeness condition even if rounded child spheres extend slightly
outside parent spheres. Any failed sphere check uses the original BVH.

Construction uses the actual packed FP32 triangle vertices, FP64 filter arithmetic
and the same frozen outward margin `128*eps32*(1+max_abs_shape_coordinate)`.
This remains an engineering margin tested on bounded geometries, not a formal
roundoff proof for arbitrary ill-conditioned meshes. Near-equal nearest distances
use the original BVH to preserve its visit-order feature selection, with the
unchanged squared-distance rule `second-best <= max(1e-20, 1e-5*best)`.

The contiguous arrays store per-shape root IDs, node centers/radii, construction
half-extents, node kind/child block/list range and packed original triangle IDs.
No particle/contact-specific tree is allocated and rigid motion does not rebuild it.

## Independent geometry checks

`examples/probe_adaptive_mesh.py` retains box and concave-prism queries with cold
external points, rounded dyadic boundaries and immediate FP32 neighbors. Capacity
24 leaves these low-triangle shapes at their roots, while capacity 1 forces bounded
subdivision/fallback. A separate frozen driver,
`examples/probe_adaptive_refinement.py`, uses a 192-triangle subdivided box to force
**child-leaf queries** at capacity 24. It produces 7,625 nodes, 5,720 usable leaves,
952 interior fallback leaves and depth 6; this normal-capacity fixture does not hit
the budget. Forced capacity 1 reaches the node budget and falls back without
truncation. All probe sources, assets, policies and generated shaders are frozen
before compilation; array hashes are recorded before candidate query execution.

Across both probes, all eight shape/capacity checks pass. Each executes 11,280
candidate queries whose complete distance/normal vectors match the original GPU
query bitwise. Independent analytical box and FP64 concave distances differ by at
most 2.16e-10 m, below the fixed 2e-9 m gate. Independent exhaustive all-triangle
checks cover 5,443 normal-capacity points in the original probe and 4,357 in the
refined probe. Every nearest candidate within the fixed 1e-12 normalized-distance
tie band is retained. Fast/tie/forced-fallback paths are explicitly exercised.

These are geometry/software checks, not a CPU DEM simulation, contact-quadrature
convergence or scientific admission. Exact archives/reports are
`evidence/adaptive-mesh-query-probe-v1/` and
`evidence/adaptive-mesh-refinement-probe-v1/`.

## Actual Hostun protocol

`studies/hostun-adaptive-mesh-v1.json` freezes the same 424-grain checkpoint/library,
2 ns timestep, loading, drag, numerical gates, 16 warmups and six restored 128-step
samples per mode. Both modes use the same experimental immutable binding helper
and original linked-list gathering. The 1.05× screening gate and 2× target are
unchanged. Sources/protocol/input hashes are frozen before execution; constructed
array hashes, build cost and storage are recorded before candidate compilation and
timing. Instrumented traversal counters run separately and are excluded from timing.
Synchronized stage measurements include host overhead and are not GPU timestamps.

Reproduce serially on Metal with new evidence identities:

```sh
PYTHONPATH=src python examples/probe_adaptive_mesh.py --output evidence/NEW-PROBE
PYTHONPATH=src python examples/probe_adaptive_refinement.py --output evidence/NEW-REFINEMENT
PYTHONPATH=src python examples/study_adaptive_mesh.py \
  studies/hostun-adaptive-mesh-v1.json --output evidence/NEW-STUDY
```

## Retained depth-first result

`hostun-adaptive-mesh-v1` completes all 13 simulation comparisons successfully,
but all 7,601 diagnostic queries still use the BVH: zero usable leaf queries,
381,597 BVH visits and 220,555 triangle evaluations, identical to baseline. Median
step time is 8.182 → 7.970 ms (1.027×), below the fixed 5% screen. This small timing
change does not demonstrate leaf-list acceleration on Hostun. No promotion.

The index builds in 97.51 s and adds 490,453,880 bytes (467.73 MiB) of GPU storage.
There are 5,194,173 nodes and 3,243,577 usable list leaves, but depth-first allocation
spends the budget unevenly and does not cover the benchmark's query regions.
Fallback leaves include 758,405 depth-limit, 541,595 interior and 1,331 budget-limit
leaves. The two small shapes remain root leaves; the other 51 shapes each reach
99,993 allocated nodes. No budget or numerical gate is loosened after observing this.

## Separately frozen breadth-first allocation

`examples/adaptive_breadth.py` and `studies/hostun-adaptive-breadth-v1.json` replace
only allocation order with a FIFO queue under the same depth, candidate and node
limits. This spreads subdivision across the shape before spending nodes deeper
in individual branches. Fallback node tags distinguish budget, depth and interior
limits; a separate instrumented query records these reasons and sphere failures.
Performance eligibility additionally requires nonzero actual fast queries, so an
all-fallback result cannot qualify from timing differences alone.

`examples/probe_breadth_mesh.py` freezes the refined-box/concave checks separately.
All four shape/capacity cases pass: 11,280 candidate query vectors agree bitwise
with the original GPU query and independent distances remain within 2.16e-10 m.
The refined box exercises real child leaves, and capacity 1 exercises node-budget
fallback. Exact evidence: `evidence/adaptive-breadth-query-probe-v1/`.

The depth-first report/archive is preserved unchanged. The breadth-first study has
its own sources, protocol, array receipt, report and identity; it does not overwrite
the original experiment or reinterpret its missing coverage as success.

## Surface-only refinement variant

The broader center-to-triangle-AABB distance test can spend nodes subdividing empty
space near the surface. A third, separately frozen variant uses breadth-first
allocation but subdivides an overfull cell only when a candidate triangle AABB
intersects its actual cell box, with the same outward margin. Failing this test
selects the **complete BVH**, never a geometric no-contact result. Usable leaves
retain the same sphere/candidate completeness checks. The 24-candidate cap, depth8,
100,000-node budget, numerical limits and performance gates stay fixed.

Implementation and protocol: `examples/adaptive_surface.py`,
`examples/study_surface_mesh.py`, `studies/hostun-adaptive-surface-v1.json`.
`examples/probe_surface_mesh.py` independently freezes the child-leaf/refined-box,
concave, boundary, near-tie and forced-fallback checks before execution. Earlier
sources/reports remain unchanged under their original identities.

## Retained breadth-first result

Median step time increases **8.228 → 8.898 ms** (0.925× throughput). All 13
simulation comparisons pass, but all 7,601 diagnostic queries still use the BVH:
6,559 budget fallbacks and 1,042 interior fallbacks, with no sphere or depth
fallback in that step. Thus this result also does not qualify the fast geometry
path on Hostun. Both the fixed speed screen and nonzero-fast-coverage requirement
fail. Tree construction takes 143.03s and uses
255,923,264 bytes (244.07 MiB) of GPU buffers.
The original failed result and exact frozen sources remain unchanged.

## Final surface-only result and comparison

| Frozen variant | Baseline ms/step | Candidate ms/step | Throughput ratio | Fast specimen queries |
|---|---:|---:|---:|---:|
| Depth-first | 8.182 | 7.970 | 1.027× | 0/7,601 |
| Breadth-first | 8.228 | 8.898 | 0.925× | 0/7,601 |
| Surface-only breadth-first | 8.189 | 9.068 | 0.903× | 0/7,601 |

All 13 comparisons in each study pass unchanged motion, force, torque, normal,
friction-history and exact contact-key gates. This does **not** qualify actual
Hostun leaf-list evaluation because every diagnostic query falls back. The small
depth-first timing difference is below the screening threshold and gives no evidence
of useful fast-list coverage. Original unsuccessful reports are retained.

The final diagnostic attributes 4,376 fallbacks to the node budget and 3,225 to
non-surface/interior nodes; no query in that step fails a sphere check or reaches
the depth cutoff. BVH visits and triangle tests remain unchanged at 381,597 and
220,555. Synchronized contact-stage median increases 7.209 → 8.062 ms; this remains
a wall measurement, not a GPU timestamp. Tree build takes **138.57 s** and uploads
**311,587,616 bytes (297.15 MiB)**. There are 5,157,309 nodes and 792,576 short-list
leaves. Coverage and traversal overhead are both unresolved; a larger global tree
is not yet justified by an observed speedup.

The separate surface-only probe passes all four shape/capacity cases. Fast lists,
near ties, child traversal and forced node-budget fallback are exercised; 11,280
candidate query vectors match the original GPU query bitwise. Across all four
probe identities there are **16 passed cases and 45,120 candidate evaluations**.
These probes do not replace scientific or material admission. Production regression
was not rerun because no production source was changed.

## Scope and next useful measurement

This closes the bounded adaptive-index experiment, not every VDB-style design.
No acceptance gate, candidate cap, depth limit or node budget was loosened to
obtain a pass. Source/protocol/private-input hashes remain unchanged for every
study. The index memory includes construction half-extents retained in the uploaded
node data; compacting those is possible but would not resolve missing query coverage.

Before another larger tree, freeze a static replay of actual specimen query points
and require demonstrated coverage and complete nearest-feature candidates before
running dynamic timing. Separately measure the collector with exact-distance work
removed as an explicitly nonphysical timing diagnostic, so the maximum benefit of
replacing that work can be estimated. These are proposed next studies, not results
obtained here. Shader occupancy, memory traffic and divergence have not been
separately attributed by a hardware profiler.

Exact reports/archives: `hostun-adaptive-mesh-v1`, `hostun-adaptive-breadth-v1`,
`hostun-adaptive-surface-v1`, `adaptive-mesh-query-probe-v1`,
`adaptive-mesh-refinement-probe-v1`, `adaptive-breadth-query-probe-v1` and
`adaptive-surface-query-probe-v1`, under `evidence/`. Copy/hash integration receipt:
`evidence/adaptive-mesh-integration-v1/`. Local only; no commit or push.
