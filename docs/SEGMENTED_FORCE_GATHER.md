# Contiguous parallel force reduction experiment

A GPU prototype replaces the per-particle linked-list force walk with contiguous
adjacency and one 64-lane workgroup per particle. It gives about **1.78×** faster
steps in the dense 27-cube fixtures. It is **not promoted**: the unchanged numerical
gates still fail, and sparse Hostun gains no throughput. The retained production
solver and its previous cache qualification are unchanged by this experiment.

## Implementation and prospective controls

`examples/segmented_shader.py` generates archived Slang modules from the production
shader. Contact resolution atomically counts endpoints, a GPU exclusive scan
creates offsets, and an integer-atomic scatter writes encoded contact/endpoint
indices into contiguous segments. A workgroup traverses each segment in strides
of 64 and reduces compensated force and torque high/low parts through shared
memory. Stress, energy, dissipation, maximum depth and active timestep rows retain
the existing formulas. Wall-only linked lists remain for wall diagnostics.

There are no floating-point atomics or host contact reductions. Extra prototype
storage is `12*N + 8*contactCapacity` bytes plus scan temporaries; adjacency is
scratch, not checkpoint state. The harness is scoped to fixed-capacity,
conservative-timestep, undamped fixtures; wall, viscous, adaptive-capacity and
active-timestep qualification remains outstanding. It is not a public solver
option. Scatter ordering remains nondeterministic.

`examples/study_segmented_gather.py` freezes package/helper source, generated
shaders, protocols and private input hashes before executing an archived driver.
Each mode runs six restored 128-step trials after 16 warmup steps, in alternating
forward/reverse order. Existing state, force, torque, contact-key, normal and slip
gates are unchanged; the minimum median gain remains 5%. Reports retain failed
checks and exit nonzero. No failed report or threshold is rewritten.

Four modes isolate the architecture and normal precision change: baseline;
`precision` (face normal only); `segmented` (both); `segmented-original`
(contiguous reduction with original normals). The spinning and Hostun studies
compare repeats and each variant against baseline. The separately frozen zero-spin
study also compares all variant pairs. All study-wide equivalence decisions fail.

## Measured results on Apple M4 Max, explicit Metal

| Fixture | Baseline ms/step | Segmented, original normals | Segmented, corrected face normals |
|---|---:|---:|---:|
| Dense 27, spin [600,800,0] rad/s | 3.6095 | 2.0322 (1.776×) | 2.0155 (1.791×) |
| Dense 27, zero spin | 3.6444 | 2.0400 (1.786×) | 2.0506 (1.777×) |
| Sparse 424-particle Hostun | 8.9849 | 9.0235 (0.9957×) | 9.0275 (0.9953×) |

Dense fixtures use the previous 100 micrometre cube, 3³ grid, spacing 0.999 side,
Gauss3 refinement level 3, 1 ns timestep and 41,472 initial surface contacts.
Hostun uses the immutable prior first-contact checkpoint and shape library, with
347 candidate pairs and 956 initial contacts. Timings include scan/scatter.

The original-normal Hostun variant passes every recorded repeat and baseline
numerical comparison, but is about 0.43% slower. In the spinning dense case it
retains one normal repeatability failure. Corrected-normal modes pass spinning
repeat checks, but their baseline comparisons fail normal/history gates; torque
relative error is also about 1.11e-4 against a 1e-4 gate. The zero-spin fixture
fails torque repeatability in all four modes, including baseline; normal correction
alone does not solve the torque floor. These failures prevent promotion despite
the dense timing gain. No scientific admission is claimed.

## Independent face-gradient probe

The original nearest-face normal is computed from `p - closestPoint`, so tangential
projection rounding is divided by a shallow separation. For a triangle-face
interior, the exact gradient is the outward face normal. The isolated precision
variant tracks the closest feature and uses this normal for feature zero; the
edge/vertex path and signed scalar distance are unchanged.

`examples/probe_mesh_normals.py` compares both frozen modules against analytical
box distances and gradients at actual FP32 query coordinates, covering inside and
outside faces at normalized depths 1e-2, 1e-4, 1e-6, 1e-7 plus outside edges and
corners. Prospective absolute limits: distance 1e-6, normal component 2e-5.
In `mesh-face-normal-probe-v2`, baseline normal error is 0.597614; corrected error
is 3.08e-7. Both scalar distance errors are 3.53e-8. Corrected passes; baseline
fails. This is bounded analytical evidence, not a whole-mesh or material validation.
The invalid v1 probe passed the global resource block incorrectly; its source,
results and explanatory failure receipt remain preserved. It supplies no numerical
evidence.

## Evidence and next bounded work

Reports, logs, protocols and exact source snapshots:

- `evidence/dense27-segmented-gather-v1/`
- `evidence/hostun-segmented-gather-v1/`
- `evidence/dense27-zero-segmented-gather-v1/`
- `evidence/mesh-face-normal-probe-v1/` (invalid driver, retained)
- `evidence/mesh-face-normal-probe-v2/`
- `evidence/segmented-gather-summary-v1/` (archive checks and analysis)

To repeat a benchmark, use a new output identity, explicit Metal, serial execution:

```sh
PYTHONPATH=src python examples/study_segmented_gather.py \
  studies/dense27-segmented-gather-v1.json --output evidence/NEW-IDENTITY
```

The smallest next architecture study is canonical endpoint ordering followed by
the same compensated reduction. It should first establish repeatability without
changing normal formulas, then measure sorting overhead. A density-dependent GPU
path could avoid scan/scatter on sparse cases, but requires its own frozen study.
The mathematically justified face correction needs separate regression and
acceptance work because its history differs from the old path. Do not waive the
old failures or treat their relative torque gate as passed.

The canonical-order follow-up is now documented in
[DETERMINISTIC_FORCE_GATHER.md](DETERMINISTIC_FORCE_GATHER.md): bitwise dense
repeatability and~1.60× cached throughput, with original zero-spin baseline
comparison still failed and production integration outstanding. This later work
does not change any failure or qualification claim in the original experiment.
