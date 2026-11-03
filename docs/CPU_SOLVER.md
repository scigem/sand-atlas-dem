# CPU solver

The CPU backend executes complete DEM simulations, using the same normalized
Slang contact, friction, wall and rigid-body equations as Metal. It is explicitly
selected and never used as a fallback. The shared simulation driver provides
pressure/strain/loading stages, diagnostics, snapshots, strict state restoration,
material-name case resolution and the existing Atlas comparison export. Existing
scientific admission and canonical Atlas projection requirements remain in force.

## Run cases

```sh
.venv/bin/sand-dem doctor --backend cpu
.venv/bin/sand-dem run experiment.json --backend cpu --output runs/cpu-case
.venv/bin/sand-dem resume runs/cpu-case/checkpoint.h5 --output runs/cpu-continued
.venv/bin/sand-dem case run Hostun --backend cpu \
  --case cases/mesh-hold-smoke.json --output runs/cpu-hostun-smoke
```

`experiment.json` is an ordinary existing experiment specification. Material
names supply reviewed geometry and do not assign calibrated contact properties.
The Hostun smoke example is a two-step dilute software check, not preparation or
calibration. Existing run directories are not overwritten.

New checkpoints retain `backend=cpu`; resume uses that backend unless explicitly
overridden. Older checkpoints without backend metadata retain the Metal default.
An explicit cross-backend override is possible because the same compensated FP32
state schema is used, but long GPU/CPU trajectory agreement has not been qualified.
Backend, compiler and source identities are recorded in the run manifest. CPU
and Metal remain explicitly labelled in results and benchmark profiles.

## Run independent jobs in parallel

Each CPU case defaults to one native execution thread, with native geometry/contact
kernels and host NumPy prefix/reduction operations. Use separate processes for
independent cases. For example, four distinct seeds can run concurrently:

```sh
printf '%s\n' 1 2 3 4 | xargs -P 4 -I SEED \
  .venv/bin/sand-dem case run Hostun --backend cpu \
  --case cases/mesh-hold-smoke.json --seed SEED \
  --output runs/cpu-hostun-seed-SEED
```

Use unique output directories. Each process owns its assembly, neighbour lists,
contact histories and checkpoints. Immutable compiled kernels are cached by source,
Slang compiler, native compiler and architecture identity. Interprocess locks
protect compilation of each cache artifact. `SAND_DEM_CPU_CACHE` can select a cache
directory; the default is under the system temporary directory. Keep the process
count within available cores and RAM. CPU jobs on the same Apple Silicon machine
also share bandwidth and thermal capacity with Metal.

Large mesh cases can instead parallelize contact collection within one process:

```sh
SAND_DEM_CPU_THREADS=8 .venv/bin/sand-dem run experiment.json \
  --backend cpu --output runs/cpu-eight-cores
```

The value is an explicit positive integer (maximum 256); the default is 1.
For the lower-level API use `CPU(threads=8)`. Only mesh pair collection is
parallelized. Other kernels, SDF collection and wall collection remain serial.
Budget both the number of processes and threads per process within available
cores. Private temporary contact arrays also increase memory use. Small cases
can run slower with multiple threads; use one thread for small independent jobs.
The run backend identity records the selected contact-collection thread count.

For API use, select `Simulation(spec, backend="cpu")`. Lower-level callers can
pass `CPU()` from `sand_atlas_dem.cpu` as the shared engine's execution adapter.
The numerical engine host class remains in `gpu.py` for compatibility with
existing callers; passing the explicit CPU adapter creates no GPU device.

## Implementation and numerical behaviour

The installed Slang CPU target cannot compile GPU barriers and user atomics.
`cpu_scheduling.slang` therefore supplies serial CPU contact scheduling and
integer update functions. A private generated source replaces only those
scheduling routines, local periodic-branch subtraction and equivalent CPU BVH
query implementation. It keeps original triangle-distance arithmetic, feature
and traversal order, force laws, slip transport, damping, diagnostics, timestep limits,
compensated integration, overflow flags and rollback equations from the solver.
Original shader files are not rewritten on disk.

The CPU mesh query retains each child's original FP32 bounding-box distance on
the traversal stack, instead of recalculating it when the child is popped. The
pruning comparison still uses the current best distance, and far/near push order,
strict nearest-feature tie comparisons and the 32-entry overflow guard are
unchanged. It stores the winning triangle/feature index and reads that feature's
pseudonormal once after traversal. A no-winner query retains the original zero
pseudonormal and distance sentinel. These reuse computed values; they introduce
no approximate distance, geometry changes or contact filtering.

`cpu_parallel.hpp` supplies the optional native worker bridge. Workers run the
original collector on separate pairs using private control/output arrays. Private
arrays grow and retry locally when necessary, up to the global capacity. The
bridge concatenates results in original pair order after workers finish, so
thread scheduling does not change contact order or subsequent force accumulation.
Global capacity overflow uses the existing checked rollback/resize path. No
C++ worker exception crosses the C ABI; native allocation/execution errors fail
the run explicitly. The bridge source is included in the compiled cache identity.

Each native collector kernel owns reusable per-pair output buffers through an
explicit C ABI scratch object, released when the Python kernel is collected.
Its mutex serializes use when callers share the same kernel. Every dispatch
reconstructs contacts, counts and flags from the bound resources; only allocation
sizes are reused. Pair-index capacity hints remain safe after neighbour reordering
or rebinding another engine. Active pair storage resizes as the pair count changes,
and each local writable limit is clamped to the current global capacity. Growing
allocations independently also preserves retry safety after an allocation failure.
Profiling counters report query attempts, overflow retries and retained output
bytes. Retained buffers trade memory lifetime for fewer allocations and repeated
queries; default one-thread execution continues along the original serial path.

Mesh collection traverses the existing source-node cluster tree and exact target
mesh BVH. SDF collection visits all directed nodes. Every original sample key is
retained; a capacity overflow fails and the existing rollback/resize path retries
from the committed state. There is no contact truncation or geometry repair.
CPU traversal order can differ from GPU order; immutable contact keys identify
history independently of discovery order.

The periodic centre branch uses FP64 subtraction of stored FP32 high/residual
coordinates before returning the local FP32 branch. This avoids losing shallow
overlap information at a periodic seam. Cell storage itself remains FP32, so its
representational error still exists; verification reconstructs the actual stored
state independently. Independent tiny-mesh triangle queries are evaluated in a
normalized local frame to avoid the reference library's absolute degeneracy
tolerances.

Generated C++ parameter structs are bound by the native compiler, rather than
guessing vector alignment in Python. Clang compiles with `-O3 -fno-fast-math
-ffp-contract=off`. Slang's compatibility C API supplies CPU code generation;
it is deprecated upstream, so the pinned SlangPy/toolchain version is material.
No SlangPy CPU/GPU device is needed. The standalone CPU doctor actually compiles
and executes the arithmetic smoke kernel.

## Verification and measured scope

The dedicated CPU suite covers independent SDF resultants, closed-form mesh and
wall forces/torques, friction history, viscous contact, active and conservative
timestep guards, atomic rollback, explicit capacity growth, periodic seams,
concave signed distance, convex/concave cluster coverage against exhaustive nodes,
unequal scales and rotation, hierarchical scans and multiwall reductions, cell
strain/drag, deterministic checkpoint replay, complete runs, named-material CLI
execution/resume and Atlas-compatible exports. Its tolerances are fixed in
`tests/test_cpu_solver.py`. It does not claim sand calibration or experimental
validation.

Frozen verification is defined by `studies/cpu-solver-verification-v4.json`, with
reports under `evidence/cpu-solver-verification-v4/`: 305 tests passed, 5 skipped,
84 GPU/render cases deselected, including 23 dedicated CPU tests. Earlier reports, including
the v1 snapshot collection failure, are retained. See `CPU_FEASIBILITY.md` for the
earlier compiler/operator probes; those are historical feasibility results.

Fresh verification on 3 November (`cpu-solver-verification-v6`) passed 307 tests,
including 25 dedicated CPU cases. The additional independent analytic test follows
an elastic two-cube release through separation and free flight for 2 microseconds,
at 1 ns and 2 ns timesteps. Its fixed gates are 1 nm displacement error,
0.1% relative velocity error (1e-5 m/s absolute), 0.2% mechanical energy error
and 1e-17 kg m/s momentum error. Mid-contact checkpoint replay is bitwise equal
for the checked primary and residual state arrays.

The v6 parallel endurance workload completed seven jobs of 6,144 full mesh steps
each: 96 repeats of the original 64-step contacting window, resetting between
repeats. Four workers achieved 2,637 aggregate steps/s versus 700 for one worker
(3.77x); each group ran for about 10 seconds. This does not establish long-duration
thermal performance. The retained v5 attempt used 2,048 uninterrupted steps and
failed its fixed-contact-count assertion as the freely moving pairs released.
Its 307 regression tests passed; the failed benchmark is not relabelled a pass.
The separately archived `cpu-pair-release-diagnostic-v1` records the resulting
contact loss and pair separation. No solver equations or tolerance gates were
changed during this fresh verification.

The whole-step workload is 32 randomly oriented grains in isolated contacting
pairs, with original mesh queries and 576 face-quadrature nodes per grain. Each
worker executes three 64-step repeats. Warm-cache observations were approximately
712 steps/s for one process, with approximately 3.9x aggregate step throughput at four
processes. The frozen reports give the exact observed times. These short jobs
include full solver steps, but their per-step timing excludes startup, geometry
preparation and diagnostics. Group elapsed time records those additional costs.

This is not the dense 424-grain Hostun workload and is not a CPU/Metal comparison.
`hostun424-cpu-replay-v2` additionally executed the original 424-grain, 53-shape
Hostun checkpoint: 53,564,544 library surface nodes and about 4.42 million active
contacts. Two CPU hold steps took 76.43 seconds (about 38.2 seconds/step).
Restoring the CPU checkpoint and repeating those steps reproduced positions,
velocities, rotations, momenta, forces, torques, accumulated work/loss and keyed
contact history exactly in the reported comparison. Estimated solver allocation
was 4.27 GB, in addition to shape objects, compiler and temporary overhead.
This establishes full-inventory execution/replay, not a converged specimen.

`hostun424-cpu-profile-v1` subsequently instrumented actual queued kernel
execution for two complete steps from the same private checkpoint. Mean step
time was 38.227 seconds: mesh contact collection 37.472 seconds (98.03%),
contact resolution 0.499 seconds (1.31%), and force gathering 0.175 seconds
(0.46%). Compilation, geometry loading, force reconstruction and diagnostics
were excluded. The original and recent 1,000-step windows of the existing Metal
preparation took 0.649 and 0.643 seconds/step, respectively, implying an observed
59–60x CPU slowdown. These use the same original geometry at different trajectory
points, with different loading controls; this is an observational comparison,
not an isolated matched-device benchmark. No additional Metal work was launched.
Frozen sources, input hashes, kernel times and observed telemetry windows are
archived under `evidence/hostun424-cpu-profile-v1/`.

The baseline CPU collector uses the older eight-corner cluster bounds path,
while Metal uses precomputed pair transforms and transformed centre/extent
bounds. Contact collection is the clear optimization target; the profile does
not separate its bounding-box cost from exact mesh queries. Cached transforms,
better traversal and native SIMD were candidate improvements, with no measured
speedup assigned to them at that baseline.

The subsequent `hostun424-cpu-cached-bounds-v1` and
`hostun424-cpu-seeded-query-v1` comparisons passed their bitwise state gates but
gave only 1.0067x and 1.0002x step speedups, respectively. Neither provided a
meaningful improvement; their production changes were removed and their reports
retained. The accepted optimization is optional native pair parallelism, tested
against the original collector on the identical checkpoint with identical loading.
`hostun424-cpu-parallel-collector-v1` passed bitwise equality for every checked
state array and contact count after reconstruction and after two full steps.
Eight workers reduced mean step time from 37.58 to 11.73 seconds (3.20x).
This is a bounded two-step measurement,
not sustained scaling or sand validation. `cpu-solver-verification-v10` additionally
passed 343 tests (5 skipped, 84 Metal/render cases deselected), including CPU
fixtures at one and four threads, explicit two-pair global capacity rollback,
and deterministic multi-pair private-buffer growth. One thread remains default.

The subsequent `hostun424-cpu-buffer-reuse-v1` compares the frozen eight-worker
collector with retained native per-pair buffers, also using eight workers. All
checked state arrays and contact counts are bitwise equal after reconstruction
and after two complete steps. The fresh baseline takes 19.345 s/step (18.750,
19.941), versus 9.625 s/step (9.487, 9.762) with reuse: **2.010x**.
Collector-only means are 18.600 and 8.877 s. Initial reconstruction remains cold
and is excluded (17.629 and 19.458 s); it is not accelerated by buffer reuse.
The first candidate reconstruction records 8,065 pair-query attempts, including
2,444 capacity-overflow retries. Both timed steps record exactly 5,621 attempts
for 5,621 candidate pairs and zero local overflow retries. Retained output
allocation is 520,126,464 bytes (496.03 MiB), excluding vector metadata and
other solver/geometry allocations.

This is a short same-checkpoint comparison with the existing Metal preparation
left running, not sustained scaling. The fresh baseline is slower than the earlier
11.73 s/step observation; absolute timings vary between studies, and the earlier
3.20x and current 2.01x ratios must not be multiplied into a measured total gain.
The final source passes 344 tests, with 5 skips and 84 Metal/render deselections,
and a 6,144-step smoke workload in `cpu-solver-verification-v14`. New checks cover
cold-to-warm retry elimination, exact serial/parallel ordering and multiple engines
sharing an adapter while pair counts and global capacities change. Earlier v11
bridge validation and v12 pointer-signature failures remain archived; v13 passed
before the final allocation-failure hardening. No numerical tolerance was relaxed.

The additional **Metal-to-CPU transfer gate remains failed**: force relative L2
difference was 0.0760% and torque 0.2329%, against that study's fixed 0.2% gates.
The follow-up `hostun424-cpu-transfer-diagnostic-v1` records maximum normalized
force/torque absolute differences of 1.70e-8/1.97e-8. These are within the already
existing preparation resultant restart bounds (rtol 2e-4, atol 3e-7), but this
does not erase the separately failed relative-only gate. Some keyed contact
normals and contact membership also differ across devices. Strict cross-backend
state interchange is unqualified; homogeneous CPU restart is verified. Neither
the larger errors' origin nor their long-trajectory effect has been established.

Restoring the old FP32 branch arithmetic did not remove the transfer difference.
Explicit Slang PRECISE mode is now requested, matching Metal's request. All 26
generated solver kernels were compared before/after that compiler request: C++
is identical when only cache-path `#line` directives are removed. The explicit-mode
full-inventory rebuild reproduced the same resultants and contact counts. The
codegen comparison and failed transfer reports remain in the evidence archive.

The small-job parallel rates must not be applied to this full Hostun workload.
Sustained scaling, simultaneous full Hostun jobs, remote Linux machines and long
material responses still require their own studies. The active Metal preparation
was left on its existing frozen sources.

## CPU worker scaling (3 November 2026)

Two frozen studies execute the original 424-grain Hostun checkpoint with the
retained-buffer collector at 2, 4, 8 and 12 workers. Each block reconstructs the
same checkpoint, then executes four complete timed hold steps. Every checked
state array and contact count matches the eight-worker reference after
reconstruction and at the four-step endpoint. All 40 timed steps report zero
local buffer-overflow retries. The solver source is unchanged from the passing
344-test v14 verification; archive audits compare all 31 runtime source files.
Loading, compilation, reconstruction, state comparison and diagnostics are
excluded from step timing. No new Metal work was launched.

| Workers/block | v1 mean s/step | v2 mean s/step |
| --- | ---: | ---: |
| 2 | 21.30 | 20.87 |
| 4 | 13.51 | 12.39 |
| 8 initial | 9.55 | 10.39 |
| 12 | 8.01 | 10.42 |
| 8 repeat | 11.30 | 8.33 |

`hostun424-cpu-worker-scaling-v1` freezes order 8,2,4,12,8 and retains a separate
scratch context for each variant. Its repeated eight-worker mean drifts +18.3%,
so the apparent 12-worker improvement is not treated as a reliable gain. The
separately frozen v2 reverses intermediate order to 8,12,4,2,8 and uses one
collector/scratch throughout, changing only its validated worker count between
blocks. This holds retained output storage at 496.03 MiB instead of accumulating
five contexts. The repeated eight-worker mean nevertheless drifts -19.9%.
Memory growth alone therefore does not explain the observed timing variability.
The existing Metal preparation remains a concurrent background workload; neither
study establishes isolated hardware scaling or steady thermal performance.

Eight workers remain the practical starting point for individual large cases;
twelve provide no repeatable improvement and no additional 2x gain. Two to four
workers yield better throughput per allocated worker in these sequential tests.
That metric is an allocation proxy, not measured CPU utilization or concurrent
multi-case throughput. Independent cases duplicate substantial geometry/state
memory and compete for bandwidth; their full-case concurrency is not qualified.
The global default remains one worker for small independent cases.

At eight/twelve workers, contact collection still takes about 90–92% of the step.
The next useful work is finer native profiling of mesh-distance queries and
per-worker tail time, to distinguish query/SIMD opportunities from scheduling
imbalance before implementing either. Reports, exact runner/source snapshots,
input hashes and derived metrics are archived under
`evidence/hostun424-cpu-worker-scaling-v1/` and `v2/`. All numerical gates passed;
these are bounded execution/performance results, not scientific admission.

## Serial mesh-query hot path (3 November 2026)

`hostun424-cpu-mesh-query-profile-v1` instruments a private generated C++ collector
with one worker, leaving production source unchanged. All checked state arrays
and contact counts match the uninstrumented collector after reconstruction and
one full step. Inclusive function scopes overlap and include probe overhead;
they must not be summed or used as an optimization speed measurement.

One original Hostun step records 60,793,086 source-node tests, 44,344,144 exact
mesh queries, 5,299,417,520 box-distance evaluations, 1,370,479,203 triangle tests
and 148,565,379 nearest-feature normal fetches. There are 2,311,194,710 popped
BVH nodes. Excluding query roots, 2,266,850,566 popped bounds were already
computed when pushed: 42.8% of box evaluations are redundant. Deferring normal
reads to the final feature avoids repeated intermediate reads (about 70.2% of
those reads for this workload). Inclusive mesh-distance time is 30.08 s versus
37.61 s for node contact; containment takes 4.84 s and cluster checks 1.12 s.
The baseline has roughly 4.42 million retained contacts, so most exact queries
do not produce contacts. These measurements identify work, not physical validity.

The retained implementation caches the original FP32 box distances on the stack
and fetches the final pseudonormal after traversal. `hostun424-cpu-query-reuse-v1`
compares one worker against one worker on identical state/loading, with three
complete timed steps. Mean step time is **44.299 to 38.063 s (1.164x)**;
individual times are 43.929/44.447/44.521 s versus 37.856/38.052/38.280 s.
Every checked state array/contact count is identical after reconstruction and
after three steps. This is a bounded paired observation under the existing
background Metal workload, not a sustained speed claim or an additional 2x gain.
Do not multiply ratios from separate studies into a claimed total speedup.

A subsequent broad cull checked all existing outward-rounded hull plane groups
for points and cluster boxes. It passed 346 small regression tests in v16 and
its full-case reconstruction matched the reference state, but the full-case
trial was expensive and showed no useful preliminary timing benefit. It was
interrupted and the broad cull removed. The untouched partial report, exact
source/protocol and interruption annotation remain in
`hostun424-cpu-full-hull-cull-v1`; no completed candidate timing or final-state
admission is claimed. A larger plane loop is not automatically a cheaper filter.

Final verification v17 restores the v15 runtime byte-for-byte and keeps the new
penetrating-point culling oracle. It passes 346 tests (5 skipped, 84 Metal/render
deselected), plus 6,144 full smoke steps. Raw-byte query checks compare with the
original shader at convex/concave face, edge and vertex ties, random points and
extreme no-winner misses. Final source matches the successful serial comparison.
Only the CPU query is changed; Metal qualification remains separate.

The next rejection study should measure candidates eliminated per unit of filter
cost, using a bounded cheap enclosure or a selected plane subset with a proven
conservative error margin. Preserve exact-query contact membership and the fixed
full-state gate. Broad all-plane rejection is not retained; further worker-count
experiments are not the current priority.

The bounded follow-up `hostun424-cpu-pair-plane-v1` selects **one existing hull
plane per directed grain pair**, maximizing its signed value at the source
centre in target coordinates. Selection scans the existing six groups once per
pair, with stored order breaking ties. Each sample checks that plane before the
original directional hull filter and exact mesh query. The extra test uses
`64 * eps32 * (1 + max(abs(local)))` padding; the selected planes already enclose
the rounded mesh vertices with outward margins. No geometry resources, samples,
contact equations or original query ordering change. A plane is only a rejection
certificate; it never generates a contact or substitutes a convex contact law.
The CPU function is derived from the shared node operator with an optional plane
argument, rather than maintaining a second copy of contact physics.

Three complete serial steps average **39.189 to 32.136 s/step (1.219x throughput,
18.0% less wall time)**. Individual reference times are 40.034/39.008/38.525 s;
candidate times are 32.001/30.568/33.838 s. Every checked state array/contact count
matches after reconstruction and three steps; source and input integrity pass.
This short comparison ran under the existing background Metal workload and does
not establish sustained performance or an eight-worker speedup. It is another
incremental serial improvement, not a demonstrated 2x gain. Verification v18
passes 346 tests plus 6,144 smoke steps with the same numerical tolerances.

Separate `hostun424-cpu-pair-plane-counts-v1` instruments both native collectors
with call counters only, over one full step each. It also preserves identical
checked state/contact counts and passes source/input integrity. Source-node and
cluster checks remain 60,793,086 and 4,141,640. Directional hull calls fall from
60,793,086 to 47,301,115 (**22.2%**); exact mesh queries from 44,344,144 to
37,221,760 (**16.1%**); box-distance calls from 3,032,566,954 to 2,456,882,482
(**19.0%**); triangle tests from 1,370,479,203 to 1,068,746,502 (**22.0%**).
Instrumented times are not performance evidence. The remaining 37.2 million
queries still greatly exceed retained contacts; stronger bounded rejection is a
possible next study, with its own frozen selection budget and identical-state
gate. Current source matches both studies and verification v18 across all 31
runtime source files.
