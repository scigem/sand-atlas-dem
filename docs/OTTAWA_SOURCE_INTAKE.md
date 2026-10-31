# Ottawa 1 µm source intake

Accessed 2026-11-01 (Australia/Sydney). This is a source acquisition/crosswalk audit
for additional named-material support, not an accepted recipe or solver run.
Neither adjacent Atlas repository is modified.

The Sand Atlas metadata distinguishes `ottawa-0000008` (Ottawa sand 1µm,
244 particles) from `ottawa-0000009` (Ottawa sand 7.3µm, 3,367 particles). Their
shared short name does not establish interchangeable shape inventories, scales
or batches. The first dataset's metadata reports 1 µm per pixel and a CC BY 4.0
source license. Exact metadata/descriptor source snapshots and hashes are retained
in `ottawa1um-source-intake-v1/local-source-receipt.json`.

The public archive route is supported by the Atlas download template:
[30-facet Ottawa archive](https://data.scigem-eng.sydney.edu.au/sand-atlas/sands/ottawa-0000008/meshes_30.zip).
Its 67,666,794 bytes hash to
`2273714017391b9aa018e755a2416a09b69aacf8132ca7c777533a12ae721a9d`.
Raw bytes remain private locally. The archive has 175 STL members, totaling
155,738,300 uncompressed bytes; the descriptor table has 244 rows. Neither
inventory is silently relabelled as the other.

The first audit failed while reporting an empty first STL without bounds; the
script, freeze, traceback and failed report remain retained. The separately
identified `ottawa1um-source-intake-v2` handles empty inputs and inspects every
member, with unchanged local and external source hashes. It executes no supplied
source, Blender, geometry preparation or GPU kernel. Exact-coordinate STL welding
is decoding only; no repair is performed.

| Finding | Evidence |
| --- | --- |
| Basic supported topology | 173 members have one closed consistently wound positive-volume component and no zero-area triangles |
| Empty input | `particle_00001.stl` has zero faces and no bounds |
| Multipart input | `particle_00162.stl` has two components; it is not silently merged or repaired |
| Scale diagnostic | Under the micrometre hypothesis, worst nearest-descriptor signed-volume difference is 0.142535% |
| Assignment ambiguity | `particle_00054.stl` and `particle_00084.stl` both have descriptor 89 as their nearest volume match |
| Unmatched descriptors | 71 descriptor IDs have no nearest-volume match; the complete list is retained |

The volume diagnostic supports further investigation but cannot establish a
one-to-one identity crosswalk. For particle 54 the nearest and second relative
errors are 1.33387e-5 and 1.76866e-5; for particle 84 they are 4.16024e-6 and
8.50818e-6. Forcing a global assignment or inferring an index offset would create
an unsupported correspondence. No unit or scale is admitted by this audit.

A read-only review of the current `sand-atlas-python` generator is retained with
hashes and locators in `generation-source-review.json`. That source skips boundary
labels and removes disconnected regions upstream; its present exporter applies
`voxel_size_m * 1000` before STL export. The current code is not bound to the
historical archive, whose observed raw volumes match descriptor micrometre volumes.
Do not infer historical output units or labels from the current generator alone.

The smallest next source action is acquisition of the historical archive's
original label-to-descriptor map and export/version scale receipt, or an exact
labelled-volume/source revision that establishes them. A reviewed subset may
exclude invalid whole members explicitly, but it still needs supported identity
and scale. Vertex-link/inertia/field qualification, static contact refinement,
GPU checks and assembly studies would follow separately. No density, friction,
loading calibration, size distribution or experimental correspondence is inferred.
The named-material CLI remains unchanged and this dataset is not registered.

## Inertia fingerprint diagnostic

`evidence/ottawa1um-axis-crosswalk-v1` independently compares signed volume and
three volume-enforced inertia-ellipsoid full axes for all 173 basic-topology
supported members. The CPU-only script freezes archive/table/source/script hashes
before computation, preserves the top two candidates, and verifies unchanged
inputs afterward. No matching threshold or forced global assignment is introduced.

The [current primary SPAM definition](https://www.spam-project.dev/docs/_modules/spam/label/label.html)
documents half axes with volume enforcement enabled by default. The current Sand
Atlas generator uses that convention for its full-axis columns. All 244 descriptor
ellipsoid volumes equal their tabulated volumes within 4.441e-16 relative.
These axes consequently add shape-ratio information, not three independent size
measurements. Current source versions still do not identify the historical producer.
Downloaded source stays private and is not executed; the independent calculation
uses the classical central-inertia identity. An analytical box and its rigidly
transformed copy agree within 2.643e-14 relative.

The original volume ambiguity is separated: mesh 54's leading candidate is
descriptor 89 (maximum feature difference about 0.0335%), while mesh 84's is
descriptor 119 (about 0.0757%). This is stronger diagnostic evidence for those
pairs, without an admitted identity or scale.

The full assignment remains nonunique. Meshes 90 and 120 both rank descriptor
125 first; mesh 120 differs by up to about 2.25% across features. Mesh 98 has
no similarly close four-feature candidate: its leading candidate is descriptor
198, with about 5.00% volume difference. Volume-only candidates for meshes 98
and 120 were 133 and 155 respectively. The exact signed discrepancies and second
candidates are retained in the report. No outliers are silently dropped and no
post-observation gate is fitted. Historical label/export provenance remains the
next source dependency; the diagnostic does not register another material.

## Historical exporter source review

`evidence/ottawa-export-history-v1` reads immutable local Git blobs without
executing them or touching the producer checkout's unrelated modifications.
Commit and blob IDs, SHA-256 hashes, dates, URLs and exact active-assignment line
locators are retained. The scale expressions changed across revisions:

- `a8be484b90a5d29f43e2ffc656b94d3c5ea7ea31` (2024-12-16) multiplies the
  volume correction by the argument named `voxel_size_m`.
- `141a7785f84dc4eb58d82a3a0648e70a0b3e7c7f` (2025-01-28) uses volume
  correction alone, leaving the underlying voxel coordinate scale.
- `e902cc5d2e373cbce3e84ba977bf8eafa2b598fc` (2025-02-04) multiplies that
  argument by 1000 before the correction.

All 175 ZIP entries carry date 2025-01-21, but ZIP timestamps are not a producer
receipt and do not select a source revision. The argument name does not establish
its supplied value, Blender export-unit settings or subsequent regeneration.
These source differences reinforce the need for the original label/export
receipt rather than interpreting the archive through the present generator.
No revision, scale or material registration is admitted by this review.
