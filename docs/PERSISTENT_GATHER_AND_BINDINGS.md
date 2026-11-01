# Persistent gathering and immutable shader bindings

The useful result is **immutable shader binding reuse**, combined with the earlier
canonical gather: dense 27-cube throughput improves from 3.67 to 1.33 ms/step,
**2.77×** over the original linked-list baseline and **1.73×** over canonical gather
with ordinary dispatch. GPU kernel arithmetic and timestep are identical for the
binding comparison. This is an experimental timing result, not a fully qualified
production revision: the dense binding study retains one failed normal-repeat
check in its original linked-list variant. Its canonical binding/native comparison
is exact. Production defaults remain unchanged.

## Plan, implementation and bounded results

1. **Persistent endpoint layout:** use fixed canonical slots per particle, verify
   current contact keys on the GPU, and rebuild changed particles only. Remove the
   CSR prefix scan; skip scatter/sort writes on verified reuse. Implemented in
   `examples/persistent_gather_shader.py` and measured against both original and
   cached canonical gathering. Exact membership and numerical checks pass, but
   removing the scan is insufficient: 2.26 ms versus 2.20 ms for canonical cache in
   the first study. No promotion.
2. **Fuse contact response and reduction:** count endpoints, construct current
   history hash, validate slots, then evaluate the existing contact response
   algebra within the deterministic reduction. Each pair endpoint evaluates the
   response; only the source writes history, contact force, damping power and
   sliding count. This removes the separate response pass and subsequent force
   reads, but duplicates response calculations. All comparisons pass; it measures
   2.37 ms versus 2.32 ms for canonical cache in the same trial. No promotion.
3. **Attribute whole-step wall time:** instrument the existing single submission
   of each 128-step batch. Canonical cache spends a median 1.63 ms/step before
   submission, 0.040 ms in finish/submission and 0.568 ms after submission. The first
   category includes Python command recording, copies/readbacks and binding work;
   the last includes device wait and checks. These are wall intervals, not GPU
   timestamps. They motivate testing host binding overhead rather than attributing
   all runtime to GPU arithmetic.
4. **Reuse immutable root shader objects:** implement `examples/immutable_bindings.py`.
   Every kernel, buffer identity, nested scalar/configuration value and entrypoint
   parameter contributes to the key. Cache entries retain recursively copied input
   trees and kernel/resource references, preventing mutable caller dictionaries
   from losing earlier Python resource identities. Objects are filled once, then
   reused through documented installed SlangPy native compute-pass APIs. Buffer
   contents remain live. No submitted command buffer is replayed, no floating-point
   arithmetic changes and no unsupported indirect dispatch is used.
5. **Verify correctness and limits:** compare native and binding-reuse dispatches;
   independently probe mutable dictionaries, different output buffers, entrypoint
   parameters, changed input identity/content and reversion to an old binding.
   Preserve whole-study failure decisions. Production integration and scientific
   admission remain open.

The fixed-slot mode still remaps current contact indices through the current hash;
it is persistent **endpoint ordering**, not persistent physical contact storage.
Geometry search and new-contact discovery run unchanged every step. No force or
geometry result is reused. Full persistent contact records and conservative
geometry-region caching were not implemented in this experiment.

## Frozen benchmark results

Explicit Metal on Apple M4 Max. Each study freezes package/helper source, generated
shaders, protocol and private input hashes before execution. Six restored 128-step
samples per mode follow 16 warmup steps, in forward/reverse mode order. Existing
numerical limits, key/normal/history checks and canonical membership checks remain
unchanged. A 2× timing target is stated prospectively; it does not waive numerical
checks. No failed study is rerun merely to obtain a passing decision.

| Study | Original ms/step | Canonical native | Candidate | Outcome |
|---|---:|---:|---:|---|
| Persistent slots | 3.598 | 2.198 | 2.261 | 1.59× original; all comparisons pass; no extra gain |
| Fused slots | 3.609 | 2.317 | 2.372 | 1.52× original; all comparisons pass; slower |
| Dense immutable bindings | 3.673 | 2.287 | 1.325 | 2.77× original; 1.73× canonical; whole-study gate fails |
| Sparse Hostun, native gather + bindings | 9.015 | — | 8.249 | 1.09×; all comparisons pass |
| Sparse Hostun, canonical + bindings | 9.015 | 9.293 | 8.309 | 1.085×; all comparisons pass |

Original dense gather plus binding reuse alone measures 2.936 ms, a 1.25× gain.
Dense canonical binding reuse reduces the host-before-submit median from 1.724 to
0.726 ms/step. Its post-submit interval stays similar, 0.552 versus 0.582 ms. Thus
the additional gain is primarily host scheduling/binding work, not faster GPU
contact arithmetic. The dense study records 30,332 binding-cache hits, 94 misses
and 94 immutable entries. The sparse study uses a later helper revision that
recursively retains mutable input trees; both exact helper revisions are archived.
This retention fix does not change shader math or cache keys.

**Retained dense failure:** one `baseline-bound` linked-list repeat fails the
original normal comparison. Its force/torque errors remain small and its other
checks pass, but the report's overall `equivalence_passed` and automatic eligibility
remain false. Both canonical modes pass their repeat checks; canonical binding
reuse versus canonical native has zero motion/force/torque comparison errors,
matching keys, normals and history. This does not rewrite the failed gate or imply
that the entire study passed. The earlier zero-spin baseline torque failures also
remain unchanged.

Sparse Hostun remains geometry dominated. These measurements do not establish 2×
for real sand specimens, larger meshes, different contact evolution or other Macs.

## Independent probes

- `persistent-slot-transition-probe-v1`: all 11 cases pass exact ordering,
  membership and hit/miss checks. Includes permutation, same-count replacement,
  additions/removals, empty/reappearing contacts, full 8,192 endpoints and explicit
  rejection at 8,193. No truncation.
- `fused-slot-reduction-probe-v1`: all seven prescribed contact-row transitions
  pass exact slot membership, analytical normal-contact force and independent FP64
  body force/torque reductions. Prospective tolerances remain `rtol=2e-6`,
  `atol=1e-10`. These are isolated reduction tests, not a CPU DEM fallback.
- `immutable-binding-transition-probe-v1`: six checks pass exact FP32 analytical
  results. Multiple configurations are recorded before submission while the caller
  mutates the same dictionaries. Tests also cover live buffer-content updates,
  new buffer identity, old-root reuse, zero-length work and retained snapshot
  identity. Repeated bindings hit the cache; changed identities/scalars miss.

## Scope and remaining work

All new implementations are experiment helpers. The original shader and production
GPU dispatch code are unchanged. Fixed slots retain the checked 8,192 endpoint/body
limit and budgeted scratch; cache storage remains large. The immutable root cache
is unbounded for a long-lived changing workload, and currently qualifies only the
buffer/scalar binding types used by these experiments. Lifetime, bounded eviction,
capacity growth, retries, walls, viscosity, active timestep paths, diagnostics and
full solver regression require a separately frozen production integration study.
The 329-pass earlier cache qualification does not cover these new dispatch paths.

Next work should prioritize production-quality binding reuse and canonical gather
integration, preserving the retained failures. For a further 2× on sparse Hostun,
geometry-region caching or cooperative mesh queries remain the major targets;
these experiments do not measure either algorithm.

Reports, exact archives and logs live under the study/probe identities above in
`evidence/`. The wall profile is `dense27-gather-submission-profile-v1`, binding
studies are `dense27-immutable-bindings-v1` and `hostun-immutable-bindings-v1`.
`evidence/persistent-bindings-summary-v1/` audits all eight archives and retains
report-derived decisions. No raw source geometry is copied or published.

Run one GPU workload at a time using a fresh output identity:

```sh
PYTHONPATH=src python examples/study_immutable_bindings.py \
  studies/dense27-immutable-bindings-v1.json --output evidence/NEW-IDENTITY
```
