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
