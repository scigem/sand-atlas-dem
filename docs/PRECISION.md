# Small increments on the Metal backend

The GPU remains FP32. Source geometry, dimensional preprocessing, diagnostics on
readback and the independent reference use FP64. A compiler option alone does not
establish that the backend preserves accumulation residuals.

## Frozen failure and correction

`studies/precision-v1.json` freezes independent analytical translations, peculiar
velocity under affine deformation, diagonal cell strain and quaternion rotation.
The old implementation failed (`evidence/precision-baseline-v1/report.json`): slow
cell strain accumulated to zero instead of approximately 2e-6, and small translation
and rotation increments also vanished. This can compromise long pressure-control
runs even when force calculations pass short comparisons.

The current implementation stores a high value and an FP32 residual for position,
velocity, spatial angular momentum, quaternion components, cumulative work/losses
and accumulated diagonal logarithmic cell strain. Cell lengths are reconstructed
from the original box and accumulated logarithmic strain. Affine increments use a
stable small-argument exponential difference; rotation uses a small-angle-safe
quaternion increment. Contact centre differences and neighbour displacement checks
include position residuals. There is no FP64 GPU requirement.

Slang's requested `precise` floating-point mode did not suffice on this installed
Metal toolchain: an initial implementation lost residuals under reassociation.
Local `precise` qualifiers also generated invalid Metal syntax in this version.
These failures remain in `verification-compensated-v1.log` and
`verification-compensated-v2.log`. The final arithmetic materializes intermediate
rounding through a bitwise operation with an immutable runtime-zero control word.
The host checks that word stays zero; it is initialized and included in rollback.
This prevents the observed optimizer elimination of compensation. It is qualified
by actual device execution, not inferred from a compiler setting.

`evidence/precision-compensated-v1/report.json` passes the unchanged gates:
translation error 3.58e-12 in normalized units, peculiar velocity error 9.91e-15,
cell strain error 6.63e-8 and rotation error 2.22e-12 rad. The full suite records
24 passed in `verification-compensated-v4.log`. The tiny-increment test also restores
all residual state and repeats advancement with bitwise-identical state on this
same device. This is bounded numerical verification, not a general error bound.

## State and limitations

Current checkpoints are `sand-dem-checkpoint/3` and contain all residual arrays.
Older checkpoints /1 and /2 are rejected; inventing missing residuals would not
reproduce the original trajectory. Snapshot /2 contains position and quaternion
residuals; the VTK exporter combines them in FP64 and can still inspect snapshot /1.
A stage strain-reference reset does not reset the original cell used for cumulative
cell integration. Failed chunks restore residuals and prescribed cell rate as well
as the main state.

Contact interpolation, force evaluation, friction springs, collision decisions and
reductions retain FP32 errors. Compensation does not make those FP64 or eliminate
SDF/grid, quadrature and timestep errors. The precision tests must be rerun for any
new compiler/backend/device; no cross-device equivalence is established.
