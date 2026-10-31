# Fixed surface resolution per source grain

The CPU preprocessor selects a fixed Gauss3 subdivision level for each grain.
It requires a completed, passing, source-stable whole-library reference report,
its original protocol and archived prospective source freeze. Missing/duplicate
poses, changed inputs and failed reference convergence reject preparation.
The Hostun reference rerun is still in progress; no adaptive Hostun asset has
been admitted or assigned to the default material recipe.

For each shape, selection chooses the smallest level from 2 through 4 whose signed
force vector agrees with every supplied reference pose within `--target-error`
(default 0.05). The existing 2% finest-reference-change gate is recomputed from
the force vectors. A selected level must be strictly coarser than each reference,
so a zero self-comparison cannot qualify it. Insufficient support fails explicitly.

After the reference study passes:

```sh
mkdir -p assets/private
sand-dem geometry adaptive-quadrature \
  assets/private/hostun-30-r64-s2048.h5 \
  assets/private/hostun-r64-adaptive-v1.h5 \
  --reference evidence/hostun-library-quadrature-adaptive-v2/report.json \
  --reference-protocol evidence/hostun-library-quadrature-adaptive-v2/protocol.json \
  --target-error 0.05 > assets/private/hostun-r64-adaptive-v1.receipt.json
```

The output contains new surface nodes/areas, preserving source vertices, faces,
SDF, COM frame, radius, volume and inertia. Each shape records its level, parent
and reference hashes, selection target and worst reference error. The receipt
retains every pose error and node count. Source geometry stays private. New
sample keys cannot replace samples in an existing checkpoint or friction history.
Preprocessing runs on the CPU; DEM remains on Metal.

This is fixed preprocessing, with no runtime adaptation. Two static poses cannot
establish accuracy for all contact orientations, tangential histories or assembly
loading. Receipts retain `gpu_verified: false` and `scientific_admission: false`.
Registration and material qualification are separate.

The GPU quadrature study driver accepts either `subdivisions` or a complete
`subdivisions_by_shape` map for an asset. Coarse/fine comparison groups by asset
identity, so differently refined grains participate in the same worst-error gate.
The unchanged 10% refined-integral limit, 0.1% same-level GPU limit and twofold
pooled worst-error improvement remain required, alongside the existing fixed-2048
node force/count baseline. Levels are checked against asset provenance, and
adaptive reference sidecars remain bound to their protocol and source archive.
The full mixed Hostun GPU study awaits the accepted reference.

## Whole-library assembly inventory

An experiment can set `shape_sampling: "balanced"` to include every source shape
with counts differing by at most one. It requires at least one particle per shape.
Placement, assignment and orientation remain seeded; the dilute nonoverlap check
remains in force. Default `"random"` preserves the previous random sequence.

For the 53-shape accepted subset, 424 particles would repeat each grain eight
times. This is a numerical sampling policy, not an inferred full-sand size
distribution or validated preparation. The larger bounded assembly also avoids
the periodic-image size limit that can prevent a 212-grain specimen from reaching
a dense state with the largest grain. Pressure, loading, convergence and
discretization gates still require execution.

## Analytical end-to-end workflow check

`mixed-quadrature-box-v3` exercises the complete reference, asset preparation and
production Metal comparison on three original analytical boxes at two depths each.
The six independent CPU reference integrals use exact box distances and a closed
continuum force; their worst continuum error is 0.2518%. The selected levels are
4, 4 and 2, so this actually exercises mixed per-shape resolution. All twelve
coarse/fine GPU cases pass the unchanged 10% integral, 0.1% same-level force and
twofold pooled-improvement gates. Worst integral error falls from 20.9604% to
4.8583%. This is static workflow verification, without Hostun or assembly admission.

The original v1 reference failure (2.1136% change against a 2% limit) and v2
mixed-profile coverage failure (both boxes selected level 4) remain retained.
V3 keeps all original shapes/poses and adds a wide plate; it keeps every numerical
acceptance limit. Reference levels through eight are permitted, while selectable
production levels remain fixed at two through four and strictly below the reference.
The higher field resolution resolves the plate interior; contacts use original
triangles. Frozen protocols and archived source bytes identify each run.

## Resolve and run the full-library GPU study

After the immutable Hostun reference finishes successfully, the complete next step is:

```sh
PYTHONPATH=src python examples/adaptive_gpu_study.py \
  studies/hostun-library-gpu-quadrature-adaptive-v1.json \
  --output evidence/hostun-library-gpu-quadrature-adaptive-v1 --run
```

Omit `--run` to prepare a separate private asset and a hash-resolved GPU protocol
without starting Metal. The resolver refuses incomplete/failed references before
creating the output. It preserves all 53 grains and both depths: 106 reference
poses and 212 coarse/adaptive GPU comparisons. The original exact-count 2,048-node
baseline remains mandatory; numerical acceptance gates are unchanged.

Contact storage is resolved before GPU execution from the largest selected-level
CPU active-node count. For each two-body case, total capacity is twice that largest
count, with a minimum of 8,192 nodes. This is a memory margin, not a force/count
tolerance or permission to truncate contacts. The earlier provisional 8,192-node
capacity is insufficient for an already retained level-two pose with 21,484 nodes;
initialization rejects overflow rather than applying advancement-time growth.
The chosen capacity, reference, candidate asset, sources and protocol are frozen
before execution. Capacity or numerical failure remains a failed study.

`adaptive-gpu-workflow-box-v1` passes all twelve original analytical-box GPU cases
through this resolver; its separate pending-reference check verifies that the
actual unfinished Hostun report cannot start preparation or GPU work. The Hostun
study itself remains pending; neither software check admits a Hostun asset.
