# Status and evidence

Current qualification update: see HOSTUN_ACQUISITION_V2.md and
MECHANICS_QUALIFICATION.md. Source-scale acquisition is now resolved for a documented
53-grain subset. Seven mechanics cases pass under compensated FP32 state updates; both geometry
resolutions are prepared. The bounded 216-sphere controller case passes; Hostun
preparation remains open. Optional exact-feature mesh queries pass fixed-quadrature
force checks, while source-triangle quadrature convergence is being studied.
The latest complete Metal test run passes 166 tests, including actual Atlas request
consumers; additional
bounded CPU studies are listed below. Historical entries below retain
their original evidence identities and are superseded where explicitly stated.

2026-10-31 — research development build, Apple M4 Max (32 GPU cores, 36 GB unified
memory), Python 3.12, SlangPy 0.43.1, Metal. The solver and CLI execute actual GPU
kernels; scientific pilot acceptance remains open.

## Latest explicit-viscosity and reference results

`normal-viscosity-integration-v1` passes all 166 tests serially on Metal.
The optional normal traction viscosity is explicit in SI units for grain contacts
and each plane; it preserves separate compensated normal loss and force-evaluation
kinematics through checkpoint /5, rollback and capacity growth. The independently
frozen constant-area oscillator v2 passes at three timesteps with 4.27-fold phase
refinement and 0.3327% finest cumulative-loss error. Its v1 preflight rejection
is retained. See CONTACT_DAMPING_DESIGN.md; restitution calibration and dynamic
mesh-assembly admission remain open.

The whole-library deterministic CPU run is terminal with all 106 rows retained,
but failed its final source-integrity gate. While it ran, separate main commit
`ff7a766` changed three broadly frozen Python files (cases, CLI and simulation).
The numerical reference helpers, geometry, reference and utility files remained
byte-identical. This observation does not override the failed frozen gate.
`hostun-library-quadrature-terminal-audit-v1` retains the terminal log, exact rows,
source changes and receipt. Two rows exceed the 2% numerical finest-change gate:
grain 38 shallow (8.0767%) and grain 54 deeper (2.6276%). The grain-38 level-six
diagnostic passes; a separately frozen grain-54 level-six diagnostic runs from an
isolated immutable source copy. Full-library consolidation/GPU checks remain open.

Historical entries below describe their original executed sources and scopes.

## Evidence identities

- `evidence/verification-v3.log`: 19 tests passed before cumulative work/fabric additions.
- `evidence/verification-v4.log`: retained failed new geometry assertion; 18 passed.
  Its assumed 0.1-cell interpolation error was incorrect at a box medial-axis cusp.
  The revised test checks the entire SDF against analytical box distances at grid
  nodes, and bounds medial-axis interpolation by half a cell. GPU tolerances unchanged.
- `evidence/verification-v2.log`: original concave-force failure retained. Mesh
  proximity predicates required order-one normalization; normalization fixed the
  geometry and passed the unchanged GPU/FP64 force tolerance.
- `evidence/contact-refinement-v1.json`: independent continuum spherical surface
  integral; normal force relative error falls from 0.835% (32/2048) to 0.0482%
  (128/32768), below the frozen 2% finest-case gate. This verifies this project's
  sampled normal traction on one geometry, not Hertz theory or the full paper.
- `evidence/benchmark-10000-v1/benchmark.json`: about 4.19 ms/step, 32/512;
  estimated working allocation 0.70 GB.
- `evidence/benchmark-10000-64-2048-v1/benchmark.json`: about 10.05 ms/step,
  64/2048. `benchmark-50000-v1`: about 17.41 ms/step, 32/512.
- `evidence/laboratory-smoke-v1/`: dilute 64-sphere, 20-step software smoke bundle
  with exports; not an equilibrated soil specimen. Historical checkpoint format /1
  predates cumulative diagnostic state. Current checkpoints are /2 and reject /1.

Benchmarks are short wall-clock measurements including host encoding and boundary
synchronization; irregular grain performance is unmeasured. GPU timestamps failed
with `SLANG_E_NOT_AVAILABLE` and remain explicitly unavailable. Initial implementation
also required correcting Slang global ParameterBlock binding via `vars`; kernel
function arguments alone did not populate global resources. The device doctor and
reference tests now exercise the corrected path.

## Scientific status

Source completeness: published normal contact formulation read and identified;
complete author solver and whole-assembly numerical reproduction unavailable.
Executable status: functioning GPU solver, laboratory, restart, exporter and viewer.
Independent verification: bounded analytical geometry/contact checks and FP64 force
comparison. Long-time integration, objectivity, work diagnostics and pressure-control
qualification remain open. Experimental validation and scientific admission: absent.

The acquired Hostun geometry is deliberately excluded from accepted assets. Its
small disconnected component and unresolved legacy units block a valid shape library.
See HOSTUN_SOURCE_GAP.md. The 10k Hostun material pilot and 50k irregular-grain study
have not been completed. No calibrated properties or constitutive-model compatibility
are inferred from the synthetic smoke run.

Latest installation verification and diagnostic-version benchmark are recorded as
new evidence identities; historical reports retain their original bytes and paths.

## Installed build verification

`evidence/doctor-installed-v1.json` confirms Apple M4 Max / Metal, 257 compiled
compute elements with zero error and no fallback. `verification-installed-v2.log`
records **20 passed**, including analytical exponential drag-energy accounting.
`laboratory-installed-v3.log` rechecks the final prepared-checkpoint/export changes.
`benchmark-installed-10000-v2/benchmark.json` measures 10.10 ms/step (20 steps × 3
repeats, 10k spheres, 64/2048); estimated working allocation 706,498,400 bytes.
Its explicit source/shader hashes identify that benchmark's code. Later laboratory
changes affect checkpoint/export handling, not the measured solver kernels.

`smoke-installed-v2` and `smoke-resumed-v2` exercise current checkpoint /2 through
the installed CLI; `smoke-export-v2` contains the final synthetic CSV/JSON and
response/VTK artifacts. These small runs remain software checks only.

## Continued qualification

`mechanics-v5` passes all seven frozen cases with proper JSON boolean gates and a
successful CLI exit. `mechanics-fast-rotation-v1` demonstrates a 25-fold orientation
error reduction under timestep refinement. Buffer-growth and evidence-serialization
regressions are fixed; `verification-qualification-v1.log` records 22 passed.

`hostun-ingestion-r32-s512-v1.json` records 53 prepared grains and one rejected grain
from the 54-member static archive; every member has its source hash and descriptor
crosswalk. The prior API error and reference-mesh rejection are not overwritten.
The finer 64/2048 preprocessing job is recoverable from per-grain cached assets.

`controller-v1` failed excessive penetration at 36,800 steps, before target pressure;
all failures/checkpoints remain. `controller-v2` freezes a latched contact-approach
rate cap of 50/s while retaining the same 1% pressure, 1e-5 kinetic/elastic, 1e-3 force
imbalance and overlap gates. It is being run to a terminal result before accepting
controller qualification. No equilibrium gate has been relaxed.

## Compensated-state qualification

See [PRECISION.md](PRECISION.md) for the frozen tiny-increment regression and the
retained compiler/roundoff failures. Current checkpoints are /3; snapshots are /2.
`verification-compensated-v4.log` records 24 passed.
`precision-compensated-v1/report.json` passes all unchanged analytical gates.

`mechanics-compensated-v1/report.json` passes all seven mechanics cases;
`mechanics-fast-compensated-v1/report.json` reduces high-spin orientation error by
15.99 times under timestep refinement. Static-cell energy-proxy drift is approximately
7.6e-5 over the declared study, without claiming an exact potential. These supersede
the earlier mechanics results for the current integrator.

Both private Hostun libraries are complete: 32/512 and 64/2048. The frozen
`hostun-sdf-refinement-v1/report.json` passes the source-surface representation gates
for all 53 grains: pooled p95 distance error falls from 0.518% to 0.158% of equivalent
diameter (3.29-fold); worst fine-grain p95 is 0.330%. This is not a force-convergence
claim at overlaps smaller than these geometric errors. Mesh contact/quadrature and
assembly convergence remain open.

`mesh-benchmark-smoke-v1/benchmark.json` executes 53 isolated mesh-derived contacting
pairs, covering all 53 grains, with 81 active surface contacts. It verifies workload
construction and memory reporting, not sustained throughput or a dense assembly.
The frozen performance protocol and source-snapshot runner are available for the
10k/50k controlled-pair studies.

`controller-v2` terminated unsuccessfully at its 500k-step budget.
`controller-v3` repeats that experiment with corrected compensated state accumulation
and immutable source/input snapshots. It failed at the 500k-step budget: pressure
9999.963 Pa and force imbalance 4.13e-4 pass, but kinetic/elastic proxy 2.33e-5
exceeds 1e-5. Maximum overlap is 0.00601. No preparation is accepted. A separately
frozen v4 experiment extends only the observation budget to one million steps;
strict continuation records its parent checkpoint and rejects changes to physical
settings or convergence gates.

The subsequent source-triangle contact audit **fails** its frozen 10% field-force
error gate: five predefined Hostun grains show up to 48.39% error at fixed shallow
poses with the 64-cell field. See HOSTUN_CONTACT_ACCURACY.md. The surface-distance
pass does not qualify material contact forces. The current Hostun libraries remain
engineering assets pending field and surface-quadrature refinement.

## Sustained actual-mesh engineering workload

`hostun-controlled-pairs-performance-v1` completed both 180-second workloads with
all intended pairs active. At 32 cells/512 surface nodes: 10k particles measured
1.605 ms/step, 5,000 pairs and 7,668 surface contacts; 50k measured 4.968 ms/step,
25,000 pairs and 38,325 surface contacts. Logical owned buffers match reported
device bytes: 121,092,572 and 357,850,288 bytes. Peak process RSS is approximately
562 MB and 914 MB, respectively (same process high-water marks). Device/host
unified-memory categories are not additive physical-memory measurements.

First/last quarter median times are 1.591/1.625 ms (10k) and 4.858/4.809 ms (50k).
No steady thermal equilibrium is inferred. The workload repeatedly restores the
same isolated-pair state; reset/diagnostics are excluded from per-step time and
included in study elapsed time. These are not dense prepared specimens. Fine
resolution is being measured from an immutable snapshot of the same solver.

The optional triangle-BVH query path is under device qualification. See
MESH_QUERY.md; the SDF failure is retained, not superseded by an untested claim.

## Current controller and exact-feature query results

The independent continuation `controller-v4` **passes** at step 617,425, using the
identical parent solver sources/shaders. `acceptance-audit.json` verifies all 100
consecutive samples and checkpoint /3: maximum pressure error 9.58e-5, energy ratio
9.9991e-6, force imbalance 2.254e-4 and overlap 0.006129. A prepared 216-sphere
checkpoint exists. This qualifies that controller case only; Hostun and its
material-response study remain open. Earlier failed preparations remain failed.

The fine sustained workload completes with matched source/shader hashes to the
coarse study. At 64/2048: 10k measures 3.434 ms/step with 162,337,356 logical buffer
bytes; 50k measures 14.829 ms/step with 399,095,072 bytes. First/last quarter medians
are 3.446/3.504 ms and 14.804/14.778 ms. These imply approximately 0.95 and 4.12
hours per million steps excluding setup, reset, diagnostics/output; a million
steps at 2 ns advances 2 ms physically. The fixed isolated-pair limitations apply.

The mesh-BVH path passes analytical/concave distance checks and the corrected-pose
static Hostun force study. Worst relative force error is 0.00965%, below 0.1%, with
matching active counts. Its original excessive-penetration failure remains archived.
See MESH_QUERY.md for the separate reference identities and scope. Full-source
triangle quadrature assets are prepared but contact convergence remains unverified.

## Mesh traversal and contact-compaction qualification

`verification-mesh-cull-compact-v4.log` passes all 37 tests. The optional mesh
path now uses nearest-first triangle traversal, conservative convex-enclosure
rejection and one-pass contact collection with integer atomic reservations.
Triangle forces retain the original concave source geometry. Growth/retry and
friction restart are covered. `hostun-mesh-force-cull-compact-v1` passes all ten
unchanged static-force gates: worst relative GPU/reference error 0.00965%, with
matching active-node counts. These results do not establish surface-quadrature
convergence.

The completed pre-culling nearest-first baseline measures 0.5168 s/step at 10k and
2.5603 s/step at 50k isolated-pair particles. Owned buffer totals are 375,990,572
and 612,748,288 bytes. The fixed-order predecessor was interrupted and has no
accepted timing; its manifest and interruption receipt remain archived. The new
optimization has a separate frozen timing protocol, pending execution after the
independent CPU triangle-quadrature study finishes.

The new sustained mesh study completes successfully:
`hostun-mesh-bvh-performance-cull-compact-v1`, with independent frozen comparison
receipt `hostun-mesh-performance-comparison-v1.json`. At 10k/50k particles it measures
15.203/60.320 ms per step, speedups 34.00/42.45 over the nearest-first baseline.
Both initial and final contact counts match exactly (final 16,428/82,201), with
all 5k/25k intended pairs active. Owned buffers are 377,124,668/613,882,384 bytes.
First/last quarter medians are 15.180/15.250 ms and 60.856/58.943 ms. No thermal
steady-state claim is made. Projected million-step costs are 4.22/16.76 hours;
at 2 ns this advances 2 ms physically. Dense soil, quadrature refinement and setup/
output costs are outside that projection. The CPU quadrature study was suspended
for uncontended timing and resumed; its separate suspension receipt records this.

The installed concave-mesh CLI smoke completes all 20 steps with checkpoint /3
and snapshots (`mesh-smoke-cull-compact-v1`). This dilute example is a software
check. All numerical checkpoints remain local and are Git-ignored; historical
controller attempts account for most of the approximately 19 GiB evidence footprint.
No failed evidence has been deleted.

## Larger-quadrature execution and independent reference

The current optional mesh collector adds conservative spatial sample-cluster
rejection, preserving all original surface-node identities. The full device suite
passes 44 tests (`verification-mesh-clusters-v1.log`); the ten frozen static Hostun
force cases also pass (`hostun-mesh-force-clusters-v1`). Two additional CPU reference
checks verify analytical inside-point retention and an unchanged independent
triangle integral with outside-point rejection. The preceding sustained timing
results remain tied to their own pre-cluster source snapshot; cluster timing is
not inferred from them.

The independent CPU quadrature oracle now has a separately frozen accelerated
identity, `hostun-triangle-quadrature-culled-v1`. Fourteen overlapping refinement
levels match the original oracle exactly. The slower original execution was then
terminated; its partial result, executed source/protocol and replacement receipt
are retained. The accelerated run completed all ten poses with the same original triangle
queries, source nodes and 2% finest-change gate. Worst finest-level change is
0.3767%, passing this bounded refinement check. The 2,048-node surface integral
differs by 10.65%–229.67% from the refined triangle integral. Unsubdivided face
Gauss3 differs by up to 15.20%; two source-face subdivisions reduce the worst
observed difference to 1.323%. Subdivided GPU assets remain unimplemented, and
whole-library contact qualification remains open.

The all-shape Gauss3 functional workload completes successfully
(`hostun-gauss3-mesh-cluster-smoke-v1`): 106 particles, all 53 contacting pairs,
3,343,608 shared source nodes, 1,852 active contacts retained after advancement,
334,792,732 owned buffer bytes and maximum overlap 0.004192. Its scope is functional
execution only; concurrent CPU reference work excludes a sustained timing claim.

The final integrated device/reference suite passes all 46 tests
(`verification-mesh-clusters-reference-v1.log`). No scientific convergence gate
has been marked complete by that software result.

## Named-material cases

The CLI now registers source/scale receipts for prepared assets and resolves names
or canonical IDs with SHA-256 verification. Versioned case files compile to the
existing experiment contract; common CLI overrides and dry-run require no GPU.
Execution automatically exports CSV and explicitly synthetic JSON, retains case
provenance, and preserves failed preparation status. All 53 tests pass on Metal,
including named-case execution, initialization failure preservation and failed
preparation export. A named Hostun two-step hold completes with 106 particles; it
is an execution check, not qualified material preparation. See
`evidence/named-case-v1/` and `docs/CASE_WORKFLOW.md`. Reviewed Hostun acquisition is now implemented (see below); a direct synthetic-test
adapter in Constitutive Atlas remains open.

## Reviewed Hostun acquisition

`material prepare hostun` downloads the pinned mesh archive/descriptor CSV, checks
the entire reviewed member inventory and unique source-volume crosswalk, retains
the known whole-grain exclusion, and prepares a private hash-checked library.
First-use `case run` performs this automatically for an unregistered Hostun name.
Dry-run and invalid case inputs never trigger acquisition. Other materials require
their own source/scale recipes or manual preparation/registration.

Cache identity includes recipe, discretization, preprocessing code and runtime
versions. An exclusive writer lock and separate grain receipts support recovery.
A registry-write failure retains the completed, hash-pinned preparation report;
retry verifies and reuses the assets. The initial HTTP 403, the subsequently
successful identified-client download and the cache-retry test failure are retained
in distinct evidence directories. The corrected suite and end-to-end checks are
recorded in `evidence/material-acquisition-tests-v3/` and
`evidence/material-acquisition-final-v3/`. The full 54-member acquisition check uses
8-cell/16-sample geometry solely for functional verification; neither that asset nor
the downloaded-material hold case qualifies contact forces or material response.
The default 64-cell/2,048-sample profile still needs source-to-material scientific
qualification. The wheel includes the reviewed recipe for fresh installs.

## Explicit Atlas recorded-path adapter

The new bridge maps a completed, independently audited isotropic/triaxial DEM case
to Atlas's existing ordered replay request and an explicitly synthetic virtual-test
sidecar. It preserves the prepared strain/time origin, computes signed triaxial q
from the full tensor, retains raw observations and distinguishes engineering, log
and finite-volume measures. Case-run configurations/policies are frozen before
GPU initialization. The actual Atlas Python converter and browser request validator
pass on the bounded analytic software fixture; neither adjacent atlas is edited.
See `docs/ATLAS_BRIDGE.md` and the prospective `atlas-bridge-software-v3` evidence.
This does not execute a continuum solver, establish qualified DEM loading or add
synthetic measurement overlays to the current GUI. Scientific material comparison
and physical experimental validation remain open.

## Subdivided surface quadrature and reference-query audit

The current GPU collector uses a conservative hierarchy over reusable sample
bounds, with stable original sample keys and full-group barriers. The full suite
passes **84 tests** (`mesh-minimum-reference-tests-v1/verification.log`), including
internal-bound retention and independent analytical/exhaustive nearest-point checks.
The unchanged ten-pose 2,048-node force/count study also passes
(`hostun-mesh-force-hierarchy-v1`, worst relative error 0.00965%). These results do
not supply new sustained performance timings.

A two-subdivision Gauss3 asset contains 53,497,728 shared nodes across all 53
Hostun grains. Its source geometry, fields and mass properties are exactly
preserved, with area error <=2.22e-16 relative (`hostun-gauss3-l2-asset-v1`). The
first GPU quadrature study completes all twenty rows but fails its unchanged
0.1% same-level force-reference gate in three rows. The failed report remains
immutable despite substantial improvement against its refined integral.

The per-node audit identifies a Trimesh proximity-query tie substitution that can
choose a farther triangle. A strict FP64 minimum agrees with exhaustive source
triangle searches at the twelve largest discrepancies in each failed row; GPU
force error over retained rounded query points is <=4.40e-6 relative. The corrected
reference utility is tested. The separate ten-pose/five-level CPU reference now
passes, with worst finest-level change 0.39498%. The corresponding new twenty-row
GPU study also passes: worst same-quadrature force error 0.01504%, and level-two
error against the refined integral 1.28981% versus 15.1993% at level zero. All
original gates remain unchanged; the old failure is not converted into a pass.
The strict 2,048-node baseline retains exact counts and its 0.00965% worst force
error. The separate whole-library source-pose/reference study passes all 106 cases
over all 53 accepted grains; corresponding GPU force/count checks also pass all
106 cases, with worst relative force error 0.02588% and exact active counts.
See `HOSTUN_CONTACT_ACCURACY.md`. Whole-library quadrature,
dense-assembly, material and experimental qualification remain open.


## Deterministic inside/outside reference audit

The expanded fine-node work exposed a second, distinct reference issue: Trimesh's
ray-parity disagreement fallback can draw a random ray. Three strict equivalence
attempts (`hostun-triangle-enclosure-equivalence-v1`, `-v2`, `-v3`) failed their
unchanged 1e-12 force-equivalence gate and were stopped with all partial data and
failed reports retained. These are reference-method failures, separate from the
GPU 0.1% force gate. No failed study was converted into a pass.

`hostun-parity-audit-v1` examines sixteen disputed grain-00021 points: nine vary
under repeated unpruned classification and seven under repeated ray-pruned
classification. Exhaustive oriented solid-angle sums consistently give zero or
one, with maximum integer residual 3.54e-16. The points have positive source
surface distances, rather than being exactly on the surface.

The new reference retains bidirectional primary rays but replaces every parity
disagreement with exhaustive solid-angle winding over the original triangles.
Only values within the prospectively frozen 1e-8 of zero or positive one are
accepted; unresolved queries fail. No random fallback remains. The sixteen-point
regression passes 1,024 classifications over repeats, ordering and batch sizes
(`hostun-deterministic-inside-v1`). Separate analytical solid/cavity integrals
match exhaustive winding classification before and after conservative pruning.
The current CPU suite passes 85 tests, with one skipped and 17 Metal tests
deselected (`deterministic-reference-tests-v1/verification.log`).

The old stochastic whole-library study is interrupted after ten completed cases;
its source hashes, private partial levels and interruption receipt are retained.
The distinct `hostun-library-quadrature-deterministic-v1` now evaluates all 106
frozen poses and levels zero through four, preserving the 2% finest-change gate.
Its result is pending. The historical five-grain studies retain their original
scope and identities; the point regression does not independently requalify their
complete integrals. Full-library quadrature, GPU comparison, assembly and material
acceptance remain open.

The integrated actual-Metal suite subsequently passes 107 tests with one skipped
(`deterministic-reference-tests-v2/verification.log`), including the explicit empty
reference rejection checks. This is software verification; the full-library
reference and scientific gates above remain pending.


## Extended prepared-sphere assembly work verification

`prepared-sphere-assembly-work-v1` passes all nine actual-Metal cases: static and
drag relaxation over 100 microseconds, and cyclic affine compression/expansion over
40 microseconds, each at 10, 5 and 2.5 ns. Every case starts from the same audited
216-sphere prepared checkpoint, with frozen physical properties and histories.
Worst kinetic-plus-spring-proxy balance residual is 0.05120% (limit 2%); independent
sampled stress-power integration differs from device cell work by at most
0.0001164% of initial energy (limit 0.1%). Dissipation increments are nonnegative;
at least 559 grain pairs remain active, and maximum overlap is 0.70184%.
All three fine/middle endpoint stress/energy comparisons pass their 1% gates.
Three independent analytic pressure-volume/time-order regressions also pass.
This is sphere-assembly engineering verification, without an exact potential or
Hostun/material/experimental acceptance claim. Mesh-assembly coverage remains open.
See MECHANICS_QUALIFICATION.md and PUBLISHED_ASSEMBLY_REFERENCE.md for the distinct
published wall-boundary benchmark and source acquisition/capability gaps.


## Published benchmark source acquisition

The Feldfogel article PDF and ETH raw-data ZIP/readme are now acquired, with all
three published MD5 values verified and SHA-256 receipts retained. The independent
read-only archive inventory covers all 19 finite numeric files. Its initial
text-header parser failure remains retained; v2 freezes the exact header lines.
The Figure 4C load file is byte-identical to the displacement file; column and
initial-state assignments are unresolved. Source reproduction also requires a
generic wall-contact/reaction contract and restitution damping, which the current
periodic solver does not implement. No benchmark, calibration or experimental
acceptance is claimed by this inventory. See PUBLISHED_ASSEMBLY_REFERENCE.md.


## Isolated Metal plane-traction operator

The explicit project full-area one-sided plane operator passes
`plane-traction-operator-metal-v1`: all eighteen sphere-cap observations are
retained, and the finest 32768-node cases pass 0.1% analytical force/torque/normal
energy bounds. Worst fine force, torque and normal energy errors are 4.55e-7,
6.43e-6 and 6.51e-7 relative. Nodal continuation, separation/return, Coulomb loss
and common-frame covariance pass the frozen 1e-5 limit. Six independent scalar
surface-quadrature/work checks pass. The sandbox adapter-initialization failure is
retained separately; the host Metal execution preserves the same numerical gates.

The operator does not advance particles or implement a wall-confined laboratory.
Production collection, reactions/work, nonperiodic neighbours, transactional
restart and source-derived restitution remain open. The inspected original thesis
contact/global-drag equations do not resolve the later benchmark damping choice.
See WALL_CONTACT.md and CONTACT_SOURCE_AUDIT.md for exact derivations and scope.

## Nonperiodic domain foundation

`nonperiodic-domain-v1` passes eleven frozen Metal checks. The expanded
`nonperiodic-domain-integration-v1` passes the complete actual-Metal suite:
131 passed, one skipped. Fixed nonperiodic domains use clipped neighbour bins,
Euclidean separation and no wrapping. All six centre-domain escapes reject and
roll back the entire chunk; independent all-pairs sets, free drift, restart,
SDF/mesh interior-contact equivalence and HDF5/export boundary identity pass.
Affine rates and non-hold laboratory stages are explicitly rejected in this mode.
The new boundary boolean persists in the existing complete checkpoint /3 experiment;
no wall state has been added. Exported volume is labelled computational-domain
volume. No physical wall confinement, reactions or moving-plane work is claimed.
See WALL_CONTACT.md for the remaining production-wall integration.

## Production translating-plane laboratory

Fixed-normal planes now integrate with the GPU contact collector, friction hash,
particle gather and kick–drift–kick solver. Mesh sample hierarchies reject plane
subtrees conservatively. Per-plane reactions and common-origin moments use integer
contact counts and multilevel GPU reductions without floating force atomics.
Compensated imposed-wall work is separate from cell work and participates in the
laboratory proxy balance. Optional explicit axis-aligned wall gaps define volume
and engineering strains while the computational domain stays fixed.
Wall state joins transactional rollback, automatic capacity growth and checkpoint /4;
wall snapshots /3 export through the existing particle VTK path.

`wall-boundaries-metal-v1` passes ten frozen checks.
`wall-boundaries-integration-v1` passes 143 tests with one skipped, including
257-body/two-wall reductions, mixed grain/wall histories and full memory inventory.
The subsequent `wall-friction-return-v1` adds independent closed cap integrals for
production partial-slip force, torque and Coulomb return loss at unchanged 0.1%
gates; its full serial actual-Metal suite passes 144 tests, one skipped.
`wall-cli-smoke-v1` pauses at step 10, resumes to step 20, retains positive contacts
at every diagnostic observation, reports positive wall work and zero cell work,
and exports CSV/JSON/VTK with explicit wall-boundary/measurement-volume identity.
The one-sphere CLI case is a software check only. Restitution, wall servos,
long assembly convergence, benchmark reconstruction and material acceptance remain
open; no earlier failed or partial Hostun quadrature study is superseded.

`wall-final-integration-v1` is the final reviewed wall build: 145 tests passed,
one skipped on Metal. Immutable checkpoint-normal identity and normalized-FP32
velocity bounds are checked before mutation. An unintended profiling-time reduction
buffer reallocation was removed. Earlier study source snapshots/results remain
unchanged; the final identity repeats the complete suite and identical CLI smoke.

## Damping scaling audit and targeted reference refinement

`restitution-scaling-audit-v1` independently verifies the ordinary linear-oscillator
restitution relation (four endpoint errors <=1.16e-12, limit 1e-10), diagnoses
sqrt(node-count) aggregate damping when a whole body mass is reused at every node,
and derives area-partition/contact-fraction and spherical-cap consequences.
`restitution-scaling-tests-v1` passes eight CPU parameter/quadrature checks.
The later author thesis is acquired privately with a source receipt and exact
locators; its mass and printed-logarithm conventions do not establish the
Feldfogel benchmark's traction damping prescription. CONTACT_DAMPING_DESIGN.md
records the explicit viscosity contract to implement. The Engine is unchanged
and still has no contact restitution dashpot.

The running deterministic whole-library reference has one completed level-four
row above its frozen 2% gate: shallow grain-00038, change 8.0767%.
`hostun-grain38-quadrature-l6-v1` repeats all seven levels at the same source pose.
Its five overlapping forces agree exactly with the parent; final change is
0.3765%, passing the unchanged reference gate. This is a targeted diagnostic,
not full-library or L2-asset admission. The original 106-pose study continues
without source mutation or case exclusion; consolidation and GPU comparison
remain open. See HOSTUN_CONTACT_ACCURACY.md.
