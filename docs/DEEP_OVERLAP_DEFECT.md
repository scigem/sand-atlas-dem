# Deep-overlap contact-law defect — 2026-11-02

**Confirmed counterexample:** the current project surface-penetration potential
can produce an attractive pair resultant for deeply overlapping convex grains.
This is an algorithmic defect in using that law for dense overlap-removal annealing.
A finer surface quadrature does not remove it. No solver force-law fix is claimed.
The physical small-overlap path is not qualified by this diagnostic either.

The earlier sign/scaling/gather and translation-energy checks remain valid: the GPU
can correctly differentiate a potential that has unacceptable deep-overlap behaviour.
Agreement with its energy gradient does not establish geometric exclusion.

## Independent exact spherical calculation

Take equal spheres of radius R, centre separation d, x=d/R, friction/viscosity zero,
and traction stiffness k (Pa/m). Both directed surface passes have half weight.
The implemented energy is E=(k/4) times the sum of the two surface integrals of
penetration squared. On one penetrating cap, with mu the polar cosine towards the
other centre, penetration is R-sqrt(R²+d²-2Rd mu), for mu>d/(2R).

For 0<x<=1, exact integration gives

```
E / (pi k R^4) = x^2/3 - x^3/4
F_separating / (pi k R^3) = 3x^2/4 - 2x/3
```

Thus F is **attractive for 0<x<8/9**. Coincidence has E=0 despite complete volume
overlap. Between coincidence and the barrier at8R/9, decreasing separation decreases
this energy. Every individual contact remains a positive compressive normal traction;
it is the geometry and resultant of these tractions that cause the failure.

For 1<=x<2, writing u=2-x and J=u³/3-u⁴/4, E/(pi k R⁴)=J/x and
F/(pi k R³)=[x(u²-u³)+J]/x². Beyond2R both vanish.
The analytic spheres are a counterexample to a universal deep-overlap exclusion
claim, not a proof of the specific trapped irregular grains' full energy landscape.

## Actual Metal mesh reproducer

Frozen identity `deep-overlap-sphere-reproducer-v1`: identical convex icospheres
(subdivision2),16384 original-mesh surface samples, R=0.5mm, k=1e10Pa/m;
static initialization only, exact mesh query, no integration or CPU simulation.
Half-weight two-pass force, frictionless construction settings and50% construction
guard retained. This synthetic mesh approximates a sphere; no measured geometry
was changed or redistributed.

| d/R | Metal separating force (N) | Exact sphere (N) |
| --- | ---: | ---: |
| 0.25 | -0.455525 | -0.470421 |
| 0.50 | -0.548787 | -0.572686 |
| 0.80 | -0.183426 | -0.209440 |
| 1.00 | 0.344457 | 0.327249 |
| 1.50 | 0.338954 | 0.372701 |
| 1.90 | 0.011198 | 0.018937 |

All six signs match the prediction. Frozen normalized force-error0.03 and
energy-error0.01 bounds pass, as do protocol/source integrity. The reproducer
succeeds at demonstrating the defect; `repulsion_requirement_met` is **false**.
No test has been weakened to claim physically acceptable repulsion.

Reproduce serially with an unused output directory:

```
PYTHONPATH=src SAND_DEM_METAL=1 .venv/bin/python \
  examples/reproduce_deep_overlap.py \
  studies/deep-overlap-sphere-reproducer-v1.json \
  --output runs/deep-overlap-sphere-reproducer-REPEAT
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_deep_overlap_counterexample.py
```

The independent regression evaluates the original normal-traction surface integral
by128-point Gaussian integration, rather than repeating the closed-form algebra.
The numeric report, frozen sources, synthetic fixture and protocol are retained in
`evidence/deep-overlap-sphere-reproducer-v1/`.

## Consequences and next implementation

Stop using the physical contact potential as a general deep-overlap untangler.
Dense annealing v3/v4 remain failed, and the refinement sweep was user-cancelled
with10 completed cases retained privately; it is not a passed study. No preparation
is running. Existing source snapshots and unsuccessful studies are preserved.

A replacement construction method must enforce geometric exclusion or use a
construction-specific overlap measure that does not reward complete coincidence.
Its requirements include repulsive response over the intended overlap domain,
containment/coincident configurations, concave meshes, and a handoff to the original
physical law below its unchanged1% guard. Merely flipping nodal normals, adding
stiffness, damping harder, or clamping a pair's radial resultant would change the
law without solving general geometric exclusion. Freeze and test those requirements
on synthetic pairs before another424-grain packing. This reproducer does not select
or implement that replacement.


Literature/implementation follow-up (2026-11-02): see [CONTACT_CHOICE_LITERATURE_REVIEW.md](CONTACT_CHOICE_LITERATURE_REVIEW.md). The next bounded experiment should compare contact-normal and selection choices before replacing the complete preparation method. YADE uses sampled-surface normals in the inspected path, with explicit huge-overlap limitations; no alternative has yet been qualified here.
