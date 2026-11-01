# Conservative triangle brick experiment

**Rejected as a speedup:** actual Hostun median step time increases from **8.196 to
17.368 ms** (0.472× throughput, 2.12× elapsed cost). All 13 numerical comparisons
pass, but the instrumented specimen step takes the BVH fallback for **every one
of 7,601 queries**. This is not evidence that the accelerated path is qualified
on Hostun; fast-path correctness is exercised by the separate bounded probes.
The unchanged 1.05× screening threshold and 2× target both fail. Production code
and defaults remain unchanged.

This experiment tests a local-frame brick directory shared by all grains using the
same shape. It preserves the original triangle closest-point function, feature
pseudonormals, surface samples, history keys, timestep and contact law. It adds no
SDF interpolation or geometry repair. It is inspired by sparse-volume access, but
does not integrate NanoVDB or implement its multilevel tree format.

## Construction and query contract

`examples/brick_mesh.py` builds a 32³ directory per shape from the **actual packed
FP32 triangles**. Each cell is enclosed by a sphere of radius R centered at C.
A nearest mesh vertex gives a valid upper bound U on distance from C to the surface.
For any point P in that sphere, its nearest surface point Q satisfies
`|C-Q| <= |C-P| + |P-Q| <= R + U + R = U + 2R`.
Every potentially winning triangle must therefore have distance from C at most
U+2R. Candidate construction queries triangle centroid spheres using the maximum
triangle radius, then filters by triangle AABB distance. Both filters include
outward padding. Triangle lists contain globally packed original triangle IDs.

The radius and construction thresholds include a fixed margin of
`128*eps32*(1+maximum_absolute_shape_coordinate)`. This is a prospectively frozen
engineering margin, not a formal error bound for every ill-conditioned triangle.
The runtime checks its selected brick's sphere before trusting a list. Outside the
shape directory, a missing/empty list, or a list over 96 triangles uses the complete
original BVH. Lists are never truncated. All retained candidates are tested with
the original closest-point algebra. A near tie (`second-best <=
max(1e-20, 1e-5*best)` for squared distances) also falls back to the BVH to retain
its visit-order feature selection.

The index is immutable throughout a restored trajectory and is shared per shape,
not allocated per particle or contact. Source samples remain in their original
order. Shape motion changes the local query coordinate; it does not rebuild the
index. This prototype uses a dense directory of optional lists; compressing that
directory or introducing a sparse multilevel tree is separate work.

## Frozen studies

`studies/hostun-brick-mesh-v1.json` uses the same actual 424-grain checkpoint/library
hashes as the previous coherent-query study. Both modes use identical experimental
immutable shader bindings and original linked-list force gathering. Sources, generated
shaders, private input hashes, loading, drag, timestep, numerical gates, 16 warmups,
six restored 128-step samples per mode, 5% screening threshold and 2× target are
frozen before execution. Constructed array hashes and build costs are recorded
before candidate shader compilation and timed evaluation. A separate instrumented
step counts BVH traversal and list work; it is excluded from throughput timings.
Synchronized stage measurements include host overhead and are not GPU timestamps.

No production default, timestep, material qualification or scientific admission is
changed by the experiment. Production shader/GPU/backend files remain unchanged.

## Independent checks

`examples/probe_brick_mesh.py` freezes source/asset identities and gates before GPU
execution. Box and concave-prism shapes each exercise 2,820 points: random interior
and exterior queries, rounded cell boundaries and immediate FP32 neighbors, zero
and selected mesh vertices. Each shape runs with capacity 96 and forced-small
capacity 1. The GPU results match the original distance/normal vector bitwise at
all 11,280 evaluations. Independent analytical box and FP64 concave triangle distances
are within 2.16e-10 m of GPU distances, against the unchanged 2e-9 m limit.

For the normal-capacity index, an independent exhaustive FP64 all-triangle search
checks candidate completeness at all 3,771 covered points; every nearest triangle
within 1e-12 normalized distance of the minimum is present. The probe exercises
3,560 fast queries, 211 near-tie fallbacks, and 5,640 forced-small-capacity BVH
fallbacks. All four shape/capacity cases pass. These are bounded geometry/software
checks, not a proof for arbitrary meshes or a CPU DEM simulation.

Exact source archives and reports are under `evidence/brick-mesh-query-probe-v1/`
and `evidence/hostun-brick-mesh-v1/`. Private source geometry is referenced and
hashed rather than redistributed.

Reproduce with fresh output identities, serially on Metal:

```sh
PYTHONPATH=src python examples/probe_brick_mesh.py --output evidence/NEW-PROBE
PYTHONPATH=src python examples/study_brick_mesh.py \
  studies/hostun-brick-mesh-v1.json --output evidence/NEW-STUDY
```

## Actual Hostun result and diagnosis

| Measurement | Original BVH | Brick wrapper |
|---|---:|---:|
| Median whole step | 8.196 ms | 17.368 ms |
| Synchronized contact stage | 7.199 ms | 16.367 ms |
| BVH queries in diagnostic step | 7,601 | 7,601 |
| BVH node visits | 381,597 | 381,597 |
| BVH triangle evaluations | 220,555 | 220,555 |
| Fast list queries | 0 | 0 |

The 1,114,536-triangle, 53-shape index takes 100.35 seconds to construct and adds
66,488,664 bytes (63.41 MiB) of GPU buffers. Its directory has 1,736,704 cells;
only 85,671 have lists within the fixed 96-triangle limit. Construction and upload
are outside timed advancement. The timing shows wrapper overhead with all queries
falling back; register pressure, memory access and divergence contributions were
not separately measured.

A separately identified static diagnostic (`examples/diagnose_brick_coverage.py`,
`evidence/hostun-brick-coverage-v1/`) examines all 955 active contacts in the original
checkpoint using FP64 local coordinates and directly recomputed triangle AABB
candidate counts. Every contact lies inside its directory and enclosing sphere,
but all lists exceed capacity: **99–646 candidates**. This supports the capacity
explanation for the original contacts; it is not a trace of all 7,601 later GPU
queries. Candidate limits and acceptance gates were not increased after observing
this failure. The original study, sources and unsuccessful outcome are preserved.

A useful follow-up would refine overfull surface cells adaptively while retaining
compact lists and complete BVH fallback, rather than blindly raising list capacity.
The current BVH tests only about 29 triangles per query on average; larger linear
lists would increase triangle work. A finer *sparse* directory or tree needs its own
frozen construction-memory and query-cost study. No claim of a 2× opportunity being
established, or of all VDB-style designs being ruled out, follows from this result.

The source hash/freeze audit, exact copies into the main checkout, and unchanged
production shader/GPU/backend identities are recorded in
`evidence/brick-mesh-integration-v1/`. Local work only; no commit or push.
