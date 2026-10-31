# Stronger Hostun reference challenge and level-five candidate

Updated 2026-11-01. The original static GPU study and running preparation remain
immutable evidence. The results below extend their accuracy assessment rather
than rewrite previous reports or silently replace geometry/history identities.

## Why the extension was needed

A full-assembly surface-resolution study must have a strictly finer independent
reference at every production level. Grain 54's shallow reference originally
ended at level four. Extending it through six revealed that a passing adjacent
2% refinement-change gate does not provide a rigorous bound on all coarser levels.
The refinement is nonmonotone; level four is worse than level three at that pose.
A separate level-eight challenge then confirms the following errors:

| Production level | Force error against shallow level-eight reference |
| --- | --- |
| 2 | 26.9467% |
| 3 | 5.04175% |
| 4 | 6.72199% |
| 5 | 1.53413% |
| 6 | 0.40617% |
| 7 | 0.177512% |

Level eight serves as the reference, not a self-qualified production level.
`hostun-grain54-shallow-quadrature-l8-v1` has final change 0.177512%; the separate
`hostun-grain54-deep-quadrature-l8-v1` has final change 0.0351449%. Both retain all
seven overlapping levels exactly, with the unchanged 2% final-change and 1e-12
nonzero overlap gates. The deep extension runs in an immutable isolated source
copy, retaining the original numerical operator while development continues.

The first shallow level-six attempt failed in reporting: its unchanged zero
coarse-level force caused a 0/0 relative difference and rejected JSON NaN output.
That source, failure and all computed levels are retained. The corrected helper
requires an exactly zero baseline to reproduce exactly zero; changed zero
references fail explicitly with a JSON-safe undefined relative difference.
No epsilon normalization or relaxed tolerance is introduced. Nonzero forces keep
the original relative comparison. Six scalar overlap-contract tests pass.

`hostun-library-quadrature-consolidated-v4` retains all 106 original poses, both
new same-pose extensions and their full nested source archives. Numerical
reference dependencies remain identical. Reference acceptance and production
selection support are separate: a source-stable accepted reference is retained
even when the default production-level cap cannot satisfy the selection target.

## Existing and revised candidate identities

The existing GPU v3 asset used level three for grain 54. Its 5% preprocessing
selection target is no longer supported by the stronger shallow reference.
This does not erase its original GPU report, alter its physical parameters or
substitute samples into the running preparation.

`hostun-retained-gpu-l8-reanalysis-v1` applies the original unchanged 10% refined
integral, 0.1% same-level GPU and twofold improvement gates to all 212 retained
force observations and the stronger same-pose references. It passes: worst
selected integral error is 5.04564%, same-level GPU error remains 0.021873%, and
coarse worst error remains 100%. The exact-count fixed-node baseline stays pinned.
This is a retrospective analysis of existing GPU observations, not new GPU
execution, a restored 5% preprocessing claim, or dynamic/material qualification.
The old preparation can continue as its explicitly bounded engineering study.

The next candidate explicitly permits production level five while retaining the
5% selection target. Its selected levels are two for 51 grains, four for grain 38
and five for grain 54. `hostun-adaptive-l5-candidate-v1` contains 53,691,264 nodes,
126,720 more than the old asset (about 0.24%). Independent HDF5 comparisons find
all 53 grains' vertices, faces, fields, COM origins, inertia, volume, radius and
spacing exactly preserved. Worst selected reference error is 2.08614%.
Its exact private asset hash is retained in the public preparation/verification
receipts. The candidate remains GPU-unverified and unregistered by default.

The prospective `hostun-library-gpu-quadrature-adaptive-v4.json` policy keeps every
original pose and all numerical gates. It requires a new serial Metal execution
once the device is free. Do not replace continuing friction histories with its
new sample keys. A different production quadrature requires its own preparation
and dynamic comparison protocol.

## Explicit higher-level capability

Face quadrature now accepts levels zero through six, rejecting uint32 sample-key
overflow before allocation. Adaptive selection still defaults to maximum level
four; higher levels require explicit `maximum_level` / `--maximum-level`.
The GPU protocol validator accepts the same supported fixed levels, without
changing source-key ordering, stiffness or quadrature formulas. The candidate
above uses explicit maximum five. Level six supports a later plus-one resolution
study, subject to separate reference, resource, GPU and assembly qualification.

No new GPU execution is run concurrently with the frozen preparation. The current
software changes have CPU analytical/contract coverage; the earlier complete
Metal suite retains its original source identity. Higher-level Metal execution
and its actual memory/time costs remain pending.
