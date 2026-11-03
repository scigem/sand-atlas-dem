# Experimental sampled-surface normals

2026-11-02. Frozen studies `source-normal-contact-v1` through `v3`;
regression `source-normal-regression-v1`. This is a bounded contact-definition
experiment following [the literature audit](CONTACT_CHOICE_LITERATURE_REVIEW.md).
It is not an admitted physical model or an accepted preparation method.

## Operator and scope

Set `contact_geometry: "mesh"` and
`contact_normal: "source-surface-experimental"` on an `Experiment` to select it.
The default remains `target-gradient`. The experimental option requires zero
friction, zero normal viscosity and no planes. Configuration round trips retain
the choice; old configurations keep the existing default.

For every penetrating source surface node, retain the target signed-distance
depth, source area, half-weight directed passes and equal/opposite reactions at
the common source node. Replace the target-distance gradient with the negative
outward source-feature pseudonormal, rotated into the common frame. A second
mesh query obtains that normal; a sample farther than normalized1e-5 from its
source mesh is rejected. No geometry, area weight or contact list is repaired.
No throughput improvement is claimed for this prototype.

Near-coincident centres (normalized squared separation below1e-12) explicitly
raise error16384. The existing overlap guard rejects the scaled-containment
probe. Neither case is solved by this operator; it is not an arbitrary-overlap
untangler. The physical1% overlap acceptance remains unchanged.

The previous penetration-squared energy is still reported, explicitly labelled
as a proxy whose gradient does not generate these forces. Work residuals using
that proxy cannot establish conservative mechanics for the experimental option.
The inherited active-contact timestep check is not a new stability proof for it.

## Frozen Metal results

Actual GPU execution used explicit Metal, serially. Independent CPU triangle
queries evaluated static references only; no CPU dynamics substituted for GPU
execution. Stiffness was1e10Pa/m. Source identities, input hashes, workloads and
tolerances are retained in the evidence freezes.

Both retained coarse-quadrature trapped pairs cleared by the first diagnostic at
step600 and remained clear through4000 steps, starting with zero velocities.
Sampling was every200 steps; this does not locate the exact clearance step.

| Pair | Initial maximum overlap | At step400 | At step600 |
| --- | ---: | ---: | ---: |
|149,309|30.593%|2.232%|0%|
|163,231|27.626%|32.118%|0%|

The bounded controller retained active-bound rollback/retry failures before
advancing. This is an isolated-pair result, not a dense specimen result or an
end-to-end preparation speedup.

The source-normal operator was repulsive at all seven sphere separations from
0.25R to1.99R. The baseline was attractive at0.25R,0.5R and0.8R. At1.99R the
force change was about0.035%, below the frozen5% shallow-force gate. Global rigid
rotation and body swap checks passed, as did static force/torque references for
the two real trapped pairs. A separate synthetic concave-prism Metal regression
checks force, torque, action/reaction and rigid rotation.

**The study did not pass all qualification gates.** Independent static reference
limits were rtol3e-4 with absolute floors1e-10N and1e-14Nm. Failed cases are:

| Case | Failed quantity | Absolute error | Reference norm |
| --- | --- | ---: | ---: |
|Sphere0.25R, target gradient|torque|2.451e-10Nm|1.059e-7Nm|
|Sphere1.99R, target gradient|force|3.148e-7N|1.379e-5N|
|Sphere1.99R, target gradient|torque|1.566e-10Nm|1.111e-10Nm|
|Sphere1.99R, source surface|torque|1.033e-13Nm|1.097e-10Nm|

These failures remain unresolved; no tolerance was widened. Finite differences
of the old energy proxy also fail the frozen1e-3 force/torque gradient gate for
both pairs, with relative discrepancies near1. Consequently the v3 report has
`software_passed: false`, `physical_passed: false`, and
`scientific_admission: false`, despite successful separation.

## Evidence and remaining work

`evidence/source-normal-contact-v1/` retains the sandbox Metal-device failure;
v2 retains the unchanged containment-guard abort. V3 records containment as an
explicit unsupported-domain observation and completes the other comparisons,
without relaxing a numerical gate or guard. Raw geometry and pair states stay
private under ignored `runs/`; numerical reports and executed source snapshots
are retained in evidence.

After the study, the full serial Metal regression passed **355 tests,8 skipped**
in70.46s, with one existing pytest metadata warning. This verifies software
regression execution, independently of the failed study qualification gates.
Input hashes were checked after execution.

Before using this option for dense preparation: resolve the residual reference
errors; define and test a suitable mechanical work/energy contract; qualify
timestep refinement and behaviour for broader concave contacts; and either
support or explicitly exclude containment/coincidence in the preparation
initializer. No new assembly preparation was launched.
