# Hostun acquisition and geometry gap — 2026-10-31

Material identity comes from adjacent `sand-atlas/_data/json/hostun-0000005.json`:
Hostun angular quartz sand, contributor Max Wiebicke, 53 particles, nominal image
resolution 1 µm/pixel, source publication https://doi.org/10.1088/1361-6501/aa8dbf .
The metadata advertises CC BY 4.0; redistribution has not been used in this repository.

The inventory endpoint returned `hostun-00`. Attempts on the bulk download endpoint
returned HTTP 500; individual endpoints using `hostun-00` and `hostun-0000005` returned
HTTP 404 for the attempted grain identifier. These failures are not proof that the
entire collection is unavailable.

The exact existing website viewer path succeeded:
https://data.scigem-eng.sydney.edu.au/sand-atlas/sands/hostun-0000005/ref_particle_30.stl

SHA-256: `5483591bfd3ad901d0a0dfe67fccf5825416c5926b6a9de18e239fee645fe05d`.
Private path: `assets/private/hostun-ref-30.stl` (ignored; not a versioned input).
It contains 59,376 faces and 29,692 exact-welded vertices. Both components are closed:
59,364 faces / volume 22,427,687.3146 coordinate units³ and a 12-face component with
volume approximately 2.83e-5 units³. Coordinate extents are approximately
243.96 × 604.44 × 597.61. The importer rejected multiple components; the failed
command is retained in `evidence/hostun-geometry-32-512.log`.

The adjacent current pipeline exports some new STL meshes in mm, but that does not
establish the units of this legacy reference asset. A deliberately explicit attempted
`--unit mm` conversion was rejected before simulation; it is not an accepted scale.
No disconnected component has been removed and no plausible physical size has been
substituted.

Smallest next action: obtain a full dataset file manifest with coordinate-unit
provenance, or a source-supported mapping from this legacy reference mesh to its
image segmentation. Then review any component-removal operation as a distinct,
hashed preprocessing decision before generating accepted Hostun assets.

Source acquisition: partial. Accepted physical shape library: absent.
Executable mesh ingestion: independently demonstrated with original synthetic solids.
Numerical Hostun study / scientific admission: not performed.
