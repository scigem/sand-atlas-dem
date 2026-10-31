# Published assembly input audit

The Figure 4 benchmark in the published IJNME article and its existing private
ETH archive were rechecked on 2026-11-01. Exact source hashes, page locators and
archive members are retained in `feldfogel-input-gap-review-v1/report.json`.
No supplied solver code was executed. The archive contains 19 numerical data files
and no solver/input deck. The current ETH browser request returned 403; local
previously acquired source bytes remain hash-pinned.

The [published article](https://doi.org/10.1002/nme.7400), page 7, section 3.2 and
Figure 4, remains the applicable identity. The linked
[2022 arXiv v1](https://arxiv.org/html/2209.15431) has a different results sequence
and omits this 27-sphere benchmark. It cannot silently supply missing inputs to
the later published case. The [ETH dataset](https://doi.org/10.3929/ethz-b-000640908)
provides the raw outputs and two-page readme.

| Input | Source support / remaining gap |
| --- | --- |
| Geometry and walls | Figure 4A: 27 spheres, radius 10 mm; fixed side/bottom boundaries and translating upper boundary |
| Initial positions/orientations | Ordered arrangement shown; precise coordinates, offsets and orientation seeds absent |
| Traction stiffness | Figure 4D: 1 GPa/mm; surface refinement shown explicitly |
| Surface curves | Panel D legend has approximately 500/1000/2000/3000/4000/6000 nodes |
| Realizations | Caption specifies three slightly varied initial orientations per averaged curve |
| Raw load mapping | D's 18 columns are count-consistent with six triples, but this does not identify column order; no grouping adopted |
| Other raw-data discrepancies | B shows five curves but has 12 columns; C file duplicates displacement; remain unresolved |
| Readme parameters | Grid spacing 0.1 mm, restitution 0.5, global damping 10/s, friction 0.2, 200000 relaxation and 10000 compression steps |
| Physical time and density | Density, timestep and loading velocity/history remain absent from inspected applicable source |
| Contact damping | Restitution is specified, but its nodal mass/traction conversion is not established |

The implemented project viscosity is explicit and independently checked; it does
not infer the source's restitution law. Moving walls, wall reactions/work and
nonperiodic boundaries are implemented, but these capabilities do not resolve
source input gaps. The two directed half-weight grain passes and one-sided plane
operator remain explicit project variants.

The smallest source action is acquisition of the author's input deck, exact
realization/column map and damping convention, or a verified institutional archive
revision containing them. Sending an author request requires explicit user
instruction. Meanwhile the project can independently verify its own sphere
assembly; that separately declared study would not complete exact reproduction of
this published benchmark. Physical sand validation remains a separate requirement.
