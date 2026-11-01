# Canonical force reduction and GPU density policy

Canonical contact ordering resolves the observed dense repeatability failure in
these fixtures. Caching the verified order restores a **1.60× dense step speedup**.
This remains an experimental source generator; production dispatch is unchanged.
The original zero-spin baseline-equivalence gate remains failed, and broader
solver integration is not qualified.

## Canonical ordering and exact cache invalidation

`examples/ordered_gather_shader.py` extends the contiguous gather prototype.
A 64-lane group bitonic-sorts each particle's endpoint indices by contact source,
target, stable surface node, contact type and endpoint. Integer append order is
irrelevant. The existing compensated workgroup force/torque reduction is retained.
Sorting scratch is8192 uints (exactly32KiB); a checked per-particle endpoint count
above8192 sets failure bit16384 before loads. No endpoint is silently discarded.

The cache stores canonical contact keys and endpoint side in a fixed8192-row
uint4 array per particle, plus cached counts. Each force evaluation uses the
newly built **current** GPU contact hash to find every cached key. Count equality
and presence of every cached key are both required. A missing key or changed count
causes a full re-sort. Only ordering is cached; geometry, slip and force are always
recomputed. The contact contract requires unique keys, as does the existing
history hash. Cache key storage is128KiB per particle, about3.38MiB for27 grains
or53MiB for424 grains, in addition to CSR scratch. This fixed allocation is a
prototype limit, not a scalable production memory policy.

The first cache study failed Metal pipeline creation: adding one shared uint
made32772 bytes, exceeding the32768-byte limit. Its exact source, log, partial
samples and failure receipt remain under `dense27-zero-cached-order-v1`.
The v2 shader temporarily aliases the final sorting slot as a cache-miss flag,
keeps that endpoint's mapped index in a lane-local register, and synchronizes flag
readers before reusing scratch. The full8192 endpoint limit is preserved and tested.

## Frozen results on explicit Metal, Apple M4 Max

Each benchmark freezes package/helper source, generated shaders, input hashes and
protocol before execution; six restored128-step trials/mode follow16 warmup steps.
Forward/reverse mode ordering and all prior numerical gates remain unchanged.
All endpoint membership checks occur outside timing. Additional diagnostics test
bitwise body state, not merely tolerance agreement.

| Study | Baseline ms/step | Candidate ms/step | Result |
|---|---:|---:|---|
| Zero spin, sort every evaluation |3.6605|3.5698|1.025×; below5% speed gate |
| Zero spin, cached ordering |3.6272|2.2660|1.601×; bitwise repeatable |
| Spin[600,800,0], cached ordering |3.6016|2.2594|1.594×; all comparisons pass |
| Spin, density policy |3.6016|2.2543|1.598×; all comparisons pass |
| Sparse Hostun, cached ordering |8.9757|9.2043|2.55% slower; all comparisons pass |
| Sparse Hostun, density policy |8.9757|9.3398|4.06% slower; all comparisons pass |

Both canonical variants are bitwise repeatable across all five repeats in the
zero-spin sorting study. Cached and uncached canonical reductions agree with
zero state/force/torque errors in the cache study. All timed cached zero-spin
trials record3456 reuse hits and zero re-sorts after warmup. That high reuse is
specific to this stable contact set; the transition probe separately tests misses.

The zero-spin original baseline still fails its own repeat checks. The cached
candidate's original baseline comparison fails torque relative error2.29e-4
against the unchanged1e-4 gate, although normals, history and keys agree and
normalized motion differences are tiny. This failure is retained; exact candidate
repeatability does not retroactively pass baseline equivalence. No face-normal
formula changes are needed for the retained cached/hybrid candidates. The
face-normal correction remains a separate experiment.

The spin and Hostun density-policy studies each pass all18 recorded numerical
comparisons. Dense canonical candidates are bitwise repeatable. Sparse dispatch
uses the original linked-list order and does **not** promise bitwise repeatability.
Neither throughput nor software equivalence establishes scientific admission.

## Density selection and transition verification

The frozen GPU threshold is `activeContacts >= 64*N`, where active contacts are
surface quadrature contacts, not physical grain-pair counts. Above the threshold,
resolution counts endpoints and gathering uses canonical cached CSR order. Below
it, resolution builds original linked lists and gathering uses the original
per-thread formula. Scatter and ordering return immediately on the sparse branch;
the scan and empty dispatches remain scheduled. No new CPU count readback or
indirect dispatch is used. The global density rule is a tested prototype, not an
optimized threshold or a proven policy for mixtures of sparse and dense regions.

The predeclared sparse nonregression limit is5%; measured4.06% passes this limit.
The policy does not remove sparse overhead and is slower than always-cached in
this sample. It should not be described as a sparse speedup. The minimum dense
speedup gate remains1.05×.

Two independent GPU probes complement the dynamic fixtures:

- `order-cache-transition-probe-v1`: all11 cases pass. Exact canonical membership
  and cache hit/miss counts are checked against independently constructed sets:
  index permutation, equal-count key replacement, removal, addition, empty set,
  reappearance, full8192 endpoints,8192 cache remap and rejection at8193.
- `hybrid-switch-probe-v2`: all7 prescribed contact-row transitions pass, at191,
  192,193,191,0,192,192 contacts for3 particles. Exact endpoint membership and
  expected dense/sparse activity are checked. GPU forces and torques agree with
  independent FP64 reductions of actual FP32 contact forces and prescribed arms
  at prospective `rtol=2e-6, atol=1e-10`. Maximum normalized absolute errors are
  1.78e-15 force and3.69e-14 torque. These are reduction tests, not a CPU DEM solver.

The invalid threshold-probe v1 used the opposite centre-branch sign in its oracle
and omitted position residuals. Its driver, protocol and failure receipt are
retained. V2 uses target-minus-source and the existing residual positions; no
numerical tolerance changed. V1 supplies no valid torque evidence.

## Reproduction and remaining integration work

Benchmark drivers are `study_ordered_gather.py`, `study_cached_ordered_gather.py`
and `study_hybrid_ordered_gather.py`. Each study has a corresponding JSON protocol
under `studies/`; run serially using a fresh evidence identity:

```sh
PYTHONPATH=src python examples/study_hybrid_ordered_gather.py \
  studies/dense27-spin-hybrid-order-v1.json --output evidence/NEW-IDENTITY
```

Evidence directories use those study names; probes have the names above. Exact
archive and input audits, preserved failure classifications and report-derived
summary are under `evidence/ordered-gather-summary-v1/`. No original source archive
or failed comparison is replaced.

Production remains open: compact/budgeted ordering storage, capacity growth and
retry/rollback semantics, walls, viscosity, active-timestep paths, changing dense
contacts and larger workloads. The legacy zero-spin comparison remains unresolved
under its frozen policy. This task does not add a public solver flag, change
production numerical defaults, restart preparation, commit or push.

A later bounded study tests persistent endpoint slots, fused contact evaluation
and immutable shader bindings. Slots/fusion add no gain; binding reuse combines
with canonical gathering to measure 2.77× dense throughput, with the whole-study
legacy repeat failure retained. See
[PERSISTENT_GATHER_AND_BINDINGS.md](PERSISTENT_GATHER_AND_BINDINGS.md).
