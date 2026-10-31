# Hostun contact accuracy: retained failed gate

The frozen `studies/hostun-source-triangle-contact-v1.json` selects five grains by
volume-order indices 0, 13, 26, 39 and 52 before execution, at two prescribed
coarse-field placement depths (0.25% and 1% of equivalent diameter). All comparisons
use exactly the same physical placements, 2,048 fine surface nodes and areas. The
reference queries the original source triangles in normalized coordinates, obtaining
inside/outside and nearest-point normals independently of grid interpolation.

`evidence/hostun-source-triangle-contact-v1/report.json` **fails** its frozen 10%
maximum relative force-error gate. Fine-field errors span 4.98%–48.39%; the shallow
case for volume-order grain 39 is the worst. Some individual cases do not improve
monotonically between 32 and 64 cells. Only 1–8 nodes per directed pass are active
in these particular shallow contacts, so surface quadrature is also poorly resolved.
This study holds that quadrature fixed to isolate field/normal error; it does not
verify the continuum traction integral.

The successful source-surface distance study is narrower: it establishes decreasing
size-relative surface distance error, not accepted force accuracy at small overlap.
Consequently neither current Hostun library is accepted for a material-response
claim. Controlled-pair performance workloads remain legitimate engineering timings
provided their discretization and this accuracy limitation are explicit.

Next work is to determine field resolution and surface quadrature sufficient for
the declared overlap range, with separately frozen references and gates. Refinement
must preserve the source geometry and contact pose; increasing overlap to hide field
error would change the physical problem. Preserve the failed report. Full assembly,
loading-rate and timestep studies follow contact qualification.

The local-corner resolution diagnostic is complete at 32, 64, 128, 256 and 512
cells (`hostun-local-field-refinement-v1/report.json`). Worst force errors across
the same ten poses are 68.76%, 48.39%, 31.56%, 28.03% and 14.12%, respectively.
Only queried corners were evaluated; these are not full baked GPU assets. A 512³
FP32 scalar cube alone costs about 512 MiB per cubic shape before padding; 53 such
cubes would approach 26.5 GiB, before solver state, and still do not pass this gate.
Actual shape aspect ratios reduce that illustrative cube estimate. This motivates
an explicit triangle-query path rather than claiming uniform refinement is solved.

A separate original-triangle quadrature study now compares source-face Gauss3
rules under zero through four triangle subdivisions at fixed source poses.
`hostun-triangle-quadrature-culled-v1` retains the frozen 2% finest-level-change
gate and exact triangle/parity oracle. Its FP64 rejection planes explicitly
contain every original source vertex, so points outside their intersection have
zero contact contribution. Analytical box tests verify no interior loss and the
unchanged full triangle integral. Fourteen overlapping levels exactly match the
original unculled oracle, including forces, depths and active counts.

The slower `hostun-triangle-quadrature-v1` was interrupted after that comparison;
its partial results, original script/protocol and interruption receipt remain.
The accelerated study completed all ten poses. Every case passes the frozen 2%
finest-change gate (worst 0.3767%). Relative to its level-four triangle integral,
the 2,048-node force error ranges from 10.65% to 229.67%; exact distance evaluation
does not resolve surface integration. Unsubdivided face Gauss3 has up to 15.20%
error; two subdivisions reduce the worst observed difference to 1.323%. These
are static, frictionless comparisons over ten poses, not rigorous error bounds
or whole-library/material qualification. These historical results use the original
proximity query; the reference-query correction below requires a new study.

## Subdivided GPU quadrature and retained reference discrepancy

`hostun-gauss3-l2-asset-v1` verifies the new full 53-grain Gauss3 asset after two
uniform subdivisions (53,497,728 nodes). Original vertices, faces, SDF, mass,
inertia and scales are preserved; area error is at most 2.22e-16 relative. The
GPU collector now traverses a conservative hierarchy of sample bounds rather
than visiting every leaf block. This preserves sample coordinates, weights and
friction keys. Whole-library contact and assembly qualification remain open.

`hostun-gpu-triangle-quadrature-v1` completes all ten poses at levels zero and two
but **fails** its frozen 0.1% GPU/same-level-reference gate. Three rows across two
shallow poses exceed it, with worst error 0.1991%. Against the historical level-four
integral, worst error falls from 15.208% to 1.3012%, satisfying its separate 10%
integral gate and twofold improvement requirement. The study remains failed;
these partial improvements do not override the failed gate.

`hostun-nearest-minimum-audit-v1` traces the discrepancy to the FP64 reference
query. Trimesh 5.2's public `proximity.closest_point` can replace the true minimum
with a farther candidate when squared distances differ by less than `tol.merge`,
using face-normal alignment to select a candidate. This affects both displacement
and traction, not just the reported triangle identity. The observed excess squared
distance reaches 8.79e-9 in diameter-normalized coordinates.

The audit evaluates every retained active node in the three failing rows with a
strict FP64 minimum over independently projected candidate triangles. GPU force
error on those same rounded query points is at most 4.40e-6 relative; maximum nodal
displacement error is 7.13e-8 diameters. Exhaustive searches over every source
triangle agree exactly with candidate minima at the twelve largest legacy-query
discrepancies per row. This diagnosis does not establish that all active nodes
were retained or validate the complete integral.

The corrected CPU utility has an analytical two-plane regression with deliberately
opposed normals and a separate randomized candidate-versus-exhaustive check.
`hostun-triangle-quadrature-minimum-v1` freezes the strict-distance query and reruns
all ten original poses and all five levels. Source geometry, placements and the
2% finest-change threshold are unchanged. The original reference and failed GPU
study are retained byte for byte. The corrected CPU study now passes all ten poses;
worst level-three to level-four change is 0.39498%, below the unchanged 2% threshold.
No shader change was made in response to this reference discrepancy.

`hostun-gpu-triangle-quadrature-minimum-v1` passes all twenty GPU rows against the
separately hashed strict reference. Worst same-level GPU force error is 0.01504%,
below the unchanged 0.1% gate. Worst error against the refined integral falls from
15.1993% at level zero to 1.28981% at level two, exceeding the required twofold
improvement and passing the unchanged 10% integral gate. Maximum sampled overlap
is 1.67393%, below the unchanged 3% guard. The prior study remains failed.

One level-two shallow case has 949 FP32 contacts versus 948 in the FP64 reference.
As declared prospectively, these counts are diagnostic for boundary classification;
all force gates still apply. The separate strict 2,048-node baseline retains exact
active counts. These results cover the same ten static, frictionless poses on five
grains; they do not qualify the remaining grains, sustained performance, friction,
dynamic response, assembly convergence or physical material behaviour.

The separately frozen `hostun-fixed-node-minimum-reference-v1` recomputes the
original 2,048-node oracle at all ten unchanged poses, without repeating placement
bisection or moving the grains. Its largest force difference from the historical
oracle is 5.01e-14 relative; all per-pass active counts are unchanged.
`hostun-mesh-force-minimum-v1` passes all ten actual Metal cases against this strict
reference, with worst relative force error 9.6541e-5 (0.00965%) and exact active
counts. This baseline remains a fixed-quadrature check, not continuum convergence.
The next subdivided GPU study additionally requires this passing baseline, its
strict-query reference identity and the exact same physical pose set before any
GPU initialization, then checks the evidence hashes again after execution.

## Whole-library source-pose coverage

`hostun-library-contact-poses-minimum-v1` freezes all 53 accepted grain identities
and their original 2,048 nodes before execution. For each grain it brackets the
first approach along its body x axis with a fixed 256-interval scan, then performs
32 FP64 source-distance bisections at target sampled depths 0.0025 and 0.01
equivalent diameters. It chooses the no-deeper side, requires both bracket width
and depth shortfall <=1e-8 diameters, and retains every failed grain and private
per-grain progress. These new poses do not replace the original ten poses.

Parity classification runs before nearest-point evaluation because exterior nodes
contribute exactly zero to traction. Active nodes use the independently tested
strict FP64 distance minimum. Analytical cube and anisotropic-box fixtures check
separation, traction vectors and active counts; the initial dimensionful zero-force
assertion error and corrected vector comparison are retained separately in
`library-contact-poses-check-v1` and `-v2`. The CPU suite passes 69 tests with 17
Metal tests deselected (`library-contact-poses-check-v3`). The whole-library source
study passes all 106 cases, with maximum depth shortfall 1.93e-12 diameters and
maximum final bracket width 2.38e-12 diameters. The corresponding
`hostun-library-mesh-force-minimum-v1` GPU study passes all 106 cases: worst
relative force error is 0.02588%, below the unchanged 0.1% gate, and every active
count matches exactly. This verifies the fixed 2,048-node computation across all
53 shapes. These checks do not qualify refined surface integration,
friction, dynamics, dense specimens or material response.


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
