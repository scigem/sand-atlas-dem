# Contact-definition literature and implementation audit

2026-11-02. Evidence identity `contact-choice-literature-audit-v1`.

**Decision:** investigate the contact normal and contact selection before replacing
our entire preparation method. The demonstrated attraction is real for our chosen
operator; it does not establish that every level-set contact formulation has the
same failure. No implementation alternative has yet been qualified on our grains.

## What was actually checked

| Primary source | Verified contact choice | Implication for this project |
| --- | --- | --- |
| [Kawamoto thesis,2018](https://thesis.caltech.edu/11042/6/RYKawamoto-2018-thesis.pdf), equations3.7-3.12 | Sample one grain against the other grain's field; use the target-field gradient for a nodal normal spring and a common-point reaction. | Our target-normal direction has a source basis. This does not authorize arbitrary initial interpenetration. |
| [Feldfogel et al.,2024](https://doi.org/10.1002/nme.7400), sections2.2-2.4, equations1-8 | Replace nodal stiffness with traction stiffness times tributary area. | Area weighting addresses resolution-dependent stiffness. The inspected derivation does not establish our half-weight symmetric two-pass variant as an arbitrary-overlap construction solver. |
| [YADE contact source](https://gitlab.com/yade-dev/trunk/-/blob/0a44ad31db9fa392464fc4b6a6e5d249f6efabf5/pkg/levelSet/LevelSetIg2.cpp), pinned commit0a44ad31 | Standard convex path uses a deepest node; its normal comes from the sampled grain's own surface. The multi-contact path also uses sampled-surface normals. Source comments explicitly exclude huge overlaps. | This is a concrete alternative contact definition, with different validity limits. It is not an established arbitrary-overlap untangler. |
| [Duriez and Galusinski,2025](https://doi.org/10.1016/j.cpc.2025.109803), sections2.1,4.3 | Strictly convex, stiff-contact scope; deepest-node contact and contact-history search. | Keep its convex scope when assessing applicability to measured concave meshes. |
| [van der Haven et al.,2023](https://doi.org/10.1016/j.cma.2023.116165), sections3,4.1-4.2 | Octree overlap-volume integration, overlap centroid and averaged normals. Eq5 derives a potential-gradient formulation; the selected Eq9 volume-force law is explicitly acknowledged to differ from it. | Overlap volume is a useful independent candidate. Its paper title does not establish exact energy consistency for every implemented volume law. |
| [GeoTaichi source](https://github.com/Yihao-Shi/GeoTaichi/blob/cafe9f2a837ef985c8b55fb6f7744618703e2525/src/dem/contact/ContactKernel.py), pinned commitcafe9f2a | The inspected LS particle-particle path queries the target field and passes its gradient/normal to the selected force model. | Target-field normals are used elsewhere too. We have not tested its deep-overlap behaviour or audited every selectable contact model. |
| [Tan et al.,2024](https://doi.org/10.1002/nme.7546), sections2,3.1,4.3 | Repulsive impulses resolve approaching nodal contacts; resting contact and mass disparity require additional treatment. | Changing force integration alone does not establish a way to separate already motionless, heavily overlapping grains. |

Paper and implementation identities must stay distinct. The reviewed HAL version
of Duriez2025 prints a target-field gradient in Eq2, whereas the pinned YADE code
computes the sampled-surface normal at lines544-552 and877-883. We verified the
printed equation visually. This discrepancy is unresolved; do not silently claim
that a source-normal prototype exactly implements the paper. The standard YADE
path also chooses its smaller grain by volume; the paper's general discussion uses
surface size. Pin the actual operator under test.

YADE volume code was inspected separately: `LevelSetVolumeIg2.cpp` computes overlap
volume, an averaged normal and centroid; `VolumeElasticLaw2.cpp` applies a
volume-power force. Neither these reads nor the cited papers prove robust escape
from complete containment or coincidence. Raw PDFs and reviewed external source
files remain private; URLs, SHA256 hashes, code revisions and locators are in
`evidence/contact-choice-literature-audit-v1/receipts.json`. ParticLS' cited public
repository returned404, so no claims about its implementation are made.

## What our counterexample establishes

These are project deductions from the retained exact-sphere calculation, not claims
that the papers studied our dense initialization:

1. For deeply overlapping spheres, the nearest target surface can lie on the far
   side. A nodal force following that target normal can then pull the sampled grain
   toward the other centre. The net attraction was reproduced on actual Metal.
2. For identical spheres the two directed passes are symmetric. Removing one pass
   and restoring full weight leaves the continuum resultant unchanged. The factor
   one-half is therefore not the cause of that counterexample.
3. Using sampled-surface normals defines a different operator. For equal spheres
   its nodal reaction contributes along the sampled surface normal towards increasing
   centre separation. Every penetrating cap node has positive axial projection for
   0<d<2R, so the continuum pair resultant is repulsive over that interval. This is
   an analytical candidate, not a GPU result for that alternative. Exact coincidence
   remains directionally degenerate.
4. That normal change would break the previous proof that forces are derivatives
   of our current penetration-squared energy. Reusing the same energy diagnostic
   as a conserved potential would be unjustified. Torques and history must be
   assessed with the proposed normal, not inherited by assumption.
5. A single representative contact can erase separate contact regions on concave
   grains. A centre-branch normal can likewise destroy relevant local geometry.
   Neither is an automatic replacement for the current multi-contact operator.

## Recommended bounded next implementation

Keep the existing mesh query, Slang/Metal execution, broad phase and shape assets.
Add an explicit experimental contact-definition selector, retaining the existing
operator as the comparison baseline. First compare a sampled-surface-normal,
area-weighted multi-contact variant against the current target-normal variant.
Call it **YADE-inspired**, not an exact YADE transplant: we would retain our area
weights and symmetric passes while changing the normal construction. Use a
representative/deepest contact only as a convex-fixture comparator.

Freeze a small synthetic matrix before compiling: equal/unequal spheres across
separations, rotated convex meshes, concave disconnected contact regions, body
swap, rigid rotation, near containment and coincidence. Include resultant force,
torque, contact continuity, and translational/rotational energy-work checks.
Degenerate geometries need an explicit supported treatment or rejection. Keep
full attraction/failed-energy observations. Then retry the two retained trapped
pairs with saved poses. No424-grain preparation until that bounded study resolves
the operator's supported range.

If the inexpensive normal variant cannot supply coherent work/torque behaviour,
prototype a volume-based construction objective with consistent translation and
rotation derivatives. Use YADE's octree integration and the VLS derivation as
references, rather than assuming its selected volume-power force is an exact
potential gradient. Runtime and memory costs must be measured on our GPU; published
CPU costs are not a performance forecast for this implementation.

No solver code changed, no preparation started, no admission status upgraded.
