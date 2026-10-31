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
