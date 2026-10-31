# Published assembly reference acquisition

Updated 2026-10-31. Source bytes acquired and inventoried; benchmark reproduction
and experimental validation remain open. Earlier access and parser failures are
retained separately.

[Feldfogel et al., IJNME 125(5), e7400 (2024)](https://doi.org/10.1002/nme.7400)
is the contact-method source already identified in CONTACT_METHOD.md. Its readable
publisher HTML, section 3.2 / Figure 4, specifies 27 ordered spheres, fixed side
and bottom boundaries, and an upper moving wall. Section 3.3 / Figure 5 instead
uses interlocked blocks and a spherical indenter. These are distinct benchmarks.
The paper directs readers to ETH for supporting data.

[The associated dataset](https://doi.org/10.3929/ethz-b-000640908) was acquired after
the initial 403/429 failures. Both the 278,662-byte ZIP and 33,393-byte two-page
readme match ETH's published MD5 values. The article PDF acquired from
[CaltechAUTHORS](https://authors.library.caltech.edu/records/9c0re-55s09) also matches
that repository's MD5. Both readme pages and article page 7 / Figure 4 were rendered
and inspected. The dataset listing identifies CC BY 4.0; Caltech states Creative
Commons Attribution. Original files remain private under
`evidence/feldfogel-source-acquisition-v1/private/`.

SHA-256 receipts:

- Archive: `0648960a2ff40bab2c708399371beb7c2f81e6452368ffb075483cccbb7ac20c`.
- Readme: `2ef872c4c39f40a2a11bb2da7988667cbcc16504f1c03c58b523d5c2e5ff198c`.
- Article: `7946df872661b81445c82e1b7bf9e544290fae325b4009876112c2b9c1e44b5f`.

`feldfogel-published-data-inventory-v2` inventories 19 numeric files, all finite,
with member hashes, dimensions and ranges. V1's text-header parse failure remains
retained with its executed source; v2 declares all three observed headers exactly
before execution. No supplied script or binary was executed.

## Source-supported inputs and gaps

Figure 4 specifies sphere radius 10 mm, load in N, displacement in mm to 0.2 mm,
and adapted surface refinement at traction stiffness 1 GPa/mm. The readme page 1
specifies volume-grid spacing 0.1 mm, restitution 0.5, global damping 10 /s,
friction 0.2, and 200,000 relaxation plus 10,000 compression steps.

The Figure 4 displacement file has 101 rows. Panel D's load array has 101 rows
and 18 columns; B and E each have 12 columns. The file named for panel C loads
is byte-identical to the displacement file. The readme's general first-column
convention does not unambiguously describe the separately stored load arrays.
No load columns are averaged, relabelled or substituted. Panel B's count also
needs reconciliation with its five plotted surface-resolution curves.

Exact initial orientations/placement, density, timestep, loading velocity,
restitution implementation and curve-column identities remain unresolved.
Do not infer them from file names or plot appearance. Figures 5–6 describe a
different interlocked-block benchmark requiring its own input/geometry audit.

## Implementation implications

The current laboratory implements periodic orthogonal cells. It has neither an
explicit wall contact API nor an independently qualified wall-reaction measurement
or contact restitution damping. Global drag alone does not implement restitution.
A periodic-cell approximation would therefore be a different study and cannot
complete the published-benchmark gate. Source acquisition must establish dimensions,
initial placement/orientations, loading schedule, contact parameters, surface/grid
discretizations, damping and output units. Missing items remain gaps.

The next implementation needs generic prescribed plane-boundary contacts/reactions
and a source-derived restitution law, independently verified for force/work/history;
only then can an exact source-supported wall-confined benchmark be frozen. Keep
the project's two directed half-weight passes explicit when comparing with the
publication. Its wall counterpart needs a declared derivation rather than an
inferred weight. Missing inputs must be acquired before freezing a complete
reproduction; independently chosen numerical settings need explicit identities and
convergence studies. Physical sand validation remains a separate gate.


The isolated plane-traction kernel now passes the independently derived project
operator study in WALL_CONTACT.md. It does not add walls to the laboratory. The
inspected original contact/global-damping source and unresolved restitution mapping
are recorded separately in CONTACT_SOURCE_AUDIT.md; no coefficient is inferred.
