# Query-order test history

All force, depth and active-count assertions passed in v1. One query-count
assertion incorrectly required strictly fewer distance queries even for a box
whose exact convex enclosure had already removed every exterior point. V2 keeps
the original force/depth/count checks unchanged, requires distance queries to equal
the active count, and requires a strict reduction when exterior candidates remain.
The original test and failed log are preserved in library-quadrature-tests-v1.
No scientific refinement, GPU force, overlap or preparation gate was changed.
