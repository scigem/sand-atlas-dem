# Exact-feature mesh distance path

The optional `contact_geometry: "mesh"` path is implemented and has passed the
bounded Metal distance and static-force checks below. The existing `"sdf"` path retains its identity and
original studies. Mesh mode changes geometric query evaluation; it retains the
same two directed surface quadratures, area-weighted traction, friction law and
rigid-body integrator. It does not replace surface-quadrature convergence.

## Geometry and GPU method

Preprocess each reusable grain once into a balanced triangle bounding-volume
hierarchy. Store rounded FP32 triangles and conservative bounds, outward face
normals, edge sums of adjacent face normals and angle-weighted vertex normals.
The source triangle coordinates and topology remain unchanged. Reject degenerate
triangles, inconsistent winding, non-two-face edges and disconnected vertex fans.
A closed orientable embedded two-manifold is required; these checks alone do not
certify the absence of every geometric self-intersection.

One hierarchy is shared by all instances of a shape, with body-frame queries and
per-instance scale/orientation. Particle broad phase is unchanged. An outside
mesh bounding-box point cannot penetrate and is culled immediately. A conservative
convex enclosure supplies an additional rejection test: outward-rounded planes
are grouped by normal direction, and one group is tested per point. Passing that
test still requires querying the original concave mesh; the enclosure never
supplies a contact force. Remaining surface nodes traverse the hierarchy nearest
child first, using a 32-entry stack and point-to-box lower bounds. Preprocessing
rejects tree depth above 30; stack overflow rejects the step. The closest triangle
and its nearest face/edge/vertex determine distance and pseudonormal sign.
Hierarchy construction runs during preprocessing. Dynamic contact evaluation runs
on the GPU.

Mesh contacts use a single query/compaction pass with integer atomic reservations.
Overflow rejects the transaction, grows capacity within the memory budget and
retries from committed state. Forces use particle gathering without floating-point
atomic additions. Compact contact order may vary; friction identity remains the
source particle, target particle and stable surface-node key.

The sign construction follows J. Andreas Bærentzen and Henrik Aanæs, *Signed
Distance Computation Using the Angle Weighted Pseudonormal*, IEEE TVCG 11(3),
243–253 (2005), DOI https://doi.org/10.1109/TVCG.2005.49 . Equations (3)–(5),
printed page 245 (PDF page 4), define the feature pseudonormal and sign; an edge
uses equal weights on its two incident unit face normals, while a vertex uses
incident corner angles. Normals used for force direction are signed nearest-point
distance gradients, not smoothed vertex normals. The pseudonormal determines sign.
The distance gradient is not unique on medial loci; no smooth-force claim is made.

Source receipt: author-institution copy downloaded 2026-10-31 from
https://backend.orbit.dtu.dk/ws/portalfiles/portal/3977815/B_rentzen.pdf , SHA-256
`762f54b51781b62320271cca7d3ac6b339595f7b612b3538b0a8a246ff603be4`.
Private copy: `assets/private/publications/pseudonormal-2005.pdf`. Relevant pages were
text-extracted and visually checked. Do not redistribute the publication.

## Frozen engineering gates

The actual GPU probe tests an analytical box with face/edge/vertex and random
inside/outside points, including two packed shape identities, within 2 nm distance
error. A concave fixture compares sign and distance against independent source
triangle queries. The Hostun force study retains all ten frozen poses, exact active
sample counts and the original triangle oracle, and requires relative force error
<=0.1%. These are bounded engineering checks; field-free contact distance does not
prove quadrature, whole-assembly response, calibration or experimental validation.

## Current results and retained failure

`verification-mesh-metal-v2.log` records the full suite after adding source-triangle
quadrature. Analytical box and concave mesh distances pass.
`hostun-mesh-force-v1` remains failed: the second original pose reaches an independently
confirmed 3.2124% sampled penetration, above the unchanged 3% guard. The first case
matched the source oracle to 0.00181%; no rejected pose was accepted. Its executed
sources and failure receipt are archived in that evidence directory.

The original coarse-SDF placement labels did not bound the deepest contact of the
finer source nodes. The independent `hostun-exact-contact-poses-v1` preparation
therefore fixes the sampled maximum depth directly from original triangle queries,
using the same five grains, nodes and target depth values. `hostun-mesh-force-v2`
passes all ten resulting poses: worst relative force error 9.6541e-5 (0.00965%),
with exact active-sample counts. Its 0.1% force gate and 3% penetration guard are
unchanged. This verifies the GPU query/force computation at fixed quadrature; it
does not make 2,048 surface nodes a converged approximation of every contact patch.

`geometry face-quadrature` creates a separate hash-identified asset using centroid
or symmetric three-point Gaussian quadrature on every source triangle. Source
vertices, faces, SDF, mass and inertia are preserved. The three-point rule integrates
quadratic polynomials exactly on planar triangles; clipped contact tractions need
a separate refinement study. Sample keys are stable original-face/local-node IDs.
The private Hostun Gauss3 asset has 3,343,608 source nodes across 53 shapes.
No contact accuracy or throughput claim follows from that node count.

The current enclosure/compaction implementation passes all 37 tests in
`verification-mesh-cull-compact-v4.log`, including concave inside-point retention,
contact-capacity growth and friction restart. The independent ten-pose force
study `hostun-mesh-force-cull-compact-v1` passes the unchanged gates, with worst
relative error 0.00965% and matching active counts.

The preceding nearest-first implementation completed isolated-pair timing at
0.5168 s/step for 10k particles and 2.5603 s/step for 50k
(`hostun-mesh-bvh-performance-nearest-v1`). These are the baseline before enclosure
culling and single-pass compaction. The earlier fixed-order traversal study was
interrupted and retains its failure/interruption receipt; it supplies no timing.
Neither timing study qualifies dense specimens or quadrature accuracy.

The enclosure/compaction run now completes both unchanged sustained workloads:
15.20 ms/step at 10k and 60.32 ms/step at 50k, respectively 34.00 and 42.45 times
faster than the nearest-first baseline. Initial and final active contact counts
match exactly. Owned buffers total 377,124,668 and 613,882,384 bytes. The comparison
receipt `hostun-mesh-performance-comparison-v1.json` checks immutable source/input
hashes, physical workload settings and contact counts. Per million steps these
times project to 4.22/16.76 hours, excluding setup, reset, diagnostics/output;
at 2 ns per step that advances 2 ms physically. These are isolated-pair timings,
not a cost guarantee for dense soil or converged quadrature.

The v1–v3 contact-growth test logs are retained. Their test setup/reference was
corrected before v4: v1 changed positions outside the committed-state transaction;
v2 treated all box penetrations as face-normal contacts although some nearest
features lay on other faces; v3 omitted friction generated by normal approach
relative to those differently oriented normals. V4 creates contact through drift
and evaluates the full independent analytical-box normal/friction law. The force
tolerance remains 1e-4; these corrections do not relax the Hostun scientific gates.
