# CPU backend feasibility — 3 November 2026

Historical feasibility results. A complete CPU execution backend has since been
implemented; see [CPU_SOLVER.md](CPU_SOLVER.md) for usage, verification and limits.

The bounded CPU study supports proceeding to a CPU engine prototype. Existing
Slang contact equations can execute natively on this Apple Silicon host without
a GPU. The complete solver cannot currently compile unchanged for CPU. No CPU
DEM engine, completed CPU packing, GPU/CPU speed ratio, or scientific admission
is claimed.

This is an explicitly requested CPU experiment, not a fallback for Metal.
The ongoing preparation and queued triaxial pilot were left on their existing
frozen Metal implementation. No GPU compilation or execution was launched by
these studies. Production backend selection and defaults remain unchanged.

## Frozen studies and observed results

`studies/cpu-slang-feasibility-v1.json` freezes seven source/entry-point probes
plus an explicit CPU-device probe. The runner snapshots sources and hashes before
execution. It uses SlangPy 0.43.1 and its bundled Slang compiler
`libslang-compiler.0.2026.12.dylib` on this host. Each probe runs in a separate
process with a 60-second timeout. Failed reports and compiler logs are retained.

| Probe | Observed result |
| --- | --- |
| Explicit SlangPy CPU device | Device creation succeeds; loading the complete solver module fails |
| `normalProbe` | C++ generation succeeds |
| `planeProbe` | C++ generation succeeds |
| Solver `smoke`, selected through compiler C API | C++ generation succeeds |
| `resolveContacts`, `fillBins` | CPU compilation rejects atomic operations |
| `collectMeshFrontiers`, `blockScan` | CPU compilation rejects group barriers |

The SlangPy module loader checks other entry points in the solver module even
when selecting `smoke`. Direct compiler entry selection isolates the simple
kernel successfully. The compiler C API is used only as a compatibility probe;
it is deprecated upstream and is not a proposed production API.

`studies/cpu-contact-execution-v1.json` then freezes the generated plane kernel,
a small C++ ABI bridge, the independent reference, tolerances, inputs and workload
before native compilation and execution. System Clang compiles the generated
code using `-O3 -std=c++17 -fno-fast-math`. The ABI bridge binds generated parameter
structures and dispatches the original kernel; it contains no replacement
contact equation.

The 256 seeded SI-unit synthetic cases include penetration, separation, sticking,
sliding and active/inactive previous slip. They use the existing fixed-plane,
zero-viscosity wall variant. The FP64 reference uses `reference.nodal_force`
with doubled half-weight stiffness coefficients to match the documented
full-weight wall variant, plus independent energy, dissipation and torque algebra.
The relative tolerance was fixed at 2e-5, with separate absolute tolerances for
force/energy, slip/loss and torque/activity. All checks passed. This is local
operator verification, not grain-pair, geometry, history-transport or dynamics
verification. The normal-viscosity branch was not executed in this follow-up.

The exploratory throughput probe evaluates 262,144 fixed contacts 64 times per
worker, with one native thread per process. Inputs and old slips remain fixed
between evaluations: this is kernel throughput, not time integration. Processes
warm up and synchronize before their timed kernel loops. Process startup and
compilation are excluded.

| Workers | Aggregate contact evaluations/s | Relative to one worker |
| ---: | ---: | ---: |
| 1 | 90.0 million | 1.00 |
| 2 | 176.0 million | 1.96 |
| 4 | 328.4 million | 3.65 |

Each worker's measured loop lasted approximately 0.19–0.21 seconds. These are
single, short observations while the GPU preparation was active, not sustained
scaling measurements. Aggregate rates sum the individual worker rates; raw worker
times and checksums are retained. CPU model placement, frequency and thermal
state were not controlled. No geometry queries, neighbour search, history lookup,
force reduction, state integration or GPU baseline are included. These values
must not be compared with the packing's nodal-contact count to predict steps/s.

Both studies report frozen-source integrity passing. Evidence is under
`evidence/cpu-slang-feasibility-v1/` and `evidence/cpu-contact-execution-v1/`.

## Recommended implementation sequence

1. Define a small engine interface for initialization, advance, diagnostics,
   checkpoint and restore. Keep an explicitly selected CPU backend; retain
   explicit Metal as the existing production path. Share case definitions,
   controllers, material parameters, source identities and output contracts.
2. Put reusable pure contact and geometry math in modules that compile without
   importing GPU scheduling entry points. Keep the NumPy reference independent.
   Retain force laws, surface quadrature and mesh preprocessing exactly.
3. Implement CPU neighbour search, mesh traversal and contact collection in
   native code. Use bounded per-worker contact storage with explicit overflow
   failures. Replace GPU barriers/scans with CPU loops or prefix sums; replace
   GPU linked-list/atomic gathering with deterministic pair contributions and
   reductions. Do not emulate GPU threads one at a time across the entire solver.
4. Implement immutable contact keys, history transport, periodic cell handling,
   compensated coordinates, rigid-body integration and timestep guards. Bind
   backend/precision identity in checkpoints; qualify cross-backend restoration
   before promising interchangeable states. No silent contact truncation or
   different constitutive law to improve speed.
5. Freeze geometry-query, pair force/torque/history, energy, integration and
   restart cases. Compare independently against the existing FP64 references
   and existing accepted Metal evidence before running production CPU cases.
   Any new GPU comparisons must wait for the active Metal work to release the GPU.
6. Benchmark a representative Hostun step and short loading segment, then
   multiple complete independent cases with an explicit core/thread budget.
   Record peak resident memory, elapsed time, backend precision and completed
   cases/hour. Separate sustained kernel timings from startup and end-to-end
   timings. Reserve memory and cores for the GPU job on this shared host.

The throughput decision remains open until mesh traversal and an end-to-end step
are measured. For an aggregate speed target, if one chosen CPU worker configuration
is S times slower than the GPU and M independent workers are available, ideal
additional capacity is M/S GPU equivalents. Real measurements must account for
memory, bandwidth and thermal contention, and for the cores assigned per worker.
Remote machines were not accessed. A production native binding/build system and
portable Linux/x86 performance have not been evaluated.

Upstream context: Slang documents preliminary C++ CPU targets and restrictions on
barriers and user atomics in its [CPU target documentation](https://github.com/shader-slang/slang/blob/master/docs/cpu-target.md).
The limitations above were also observed directly in the bundled compiler;
documentation support is not treated as a successful local execution.
