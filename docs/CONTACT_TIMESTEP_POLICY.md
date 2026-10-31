# Active-contact timestep policy

The solver has an opt-in `"timestep_policy": "active-contact"` experiment setting.
It replaces the whole-surface preflight with device checks of the committed and
trial active contact networks. The default remains `"conservative"`. Specify the
setting in an experiment or a named-material case's `experiment` object; resolved
inputs and checkpoint experiment metadata retain it. `timestep` is still an explicit,
fixed number of seconds. No time, stage budget or observation interval is rescaled.

```json
{
  "contact_geometry": "mesh",
  "timestep_policy": "active-contact",
  "timestep": 1e-8
}
```

These fields belong inside a complete existing experiment/case. This is a software
example, not a Hostun timestep recommendation. After `engine.initialize()`, Python
callers can inspect `engine.timestep_limits()` for per-particle elastic and viscous
estimates in SI seconds. Zero active stiffness/viscosity gives infinity for that
component; motion checks still apply. Initialize or restore before reading limits.
The timestep is never selected or increased automatically. A rejected trial leaves
its whole advancement chunk uncommitted; choose a smaller explicit step in a new
study, preserving the old protocol and failed evidence.

## Estimate and device checks

For each active surface node attached to body i, let a be its actual common-point
lever arm, m its mass and B an upper bound on the largest eigenvalue of inverse
body inertia. B is the maximum absolute row sum of that symmetric matrix; its
eigenvalue bound is independent of the current orientation. Define mobility
`b = 1/m + |a|² B`. This includes translation, rotation and their coupling.
The receiving grain uses its own lever arm, mass and inertia.

Each body sums `g A [kn (1+mu) + kt] b` into L and
`g A eta (1+mu) b` into D. A is the node's physical quadrature area, half-weighted
for each directed grain contact and fully weighted for a plane. g=2 for a grain
pair bounds coupling between its two moving endpoints; g=1 for a fixed/prescribed
plane. Every active contact contributes, so multiple neighbours and multiple
walls add. Plane coefficients are independent of grain coefficients. The Coulomb
terms include normal-to-tangential force coupling, rather than using only a
sticking spring estimate. Full tangential stiffness is retained even when sliding.

Checks require `dt sqrt(L) <= 0.1` and `dt D <= 0.1` at every body. The factor is
fixed in this policy revision. These are conservative frozen-contact estimates,
not a computed maximum accurate timestep or a proof for the nonlinear mesh law.
Changing normals, switching closest features, friction-history transport and
contact creation can require additional refinement. A finer trajectory remains
necessary for a new loading/geometry regime.

Additional device guards limit predicted rotation to 0.01 rad, predicted surface
travel to a quarter of the neighbour skin and imposed diagonal cell log-strain
to 0.01 per step. Travel includes centre speed, radius times spin and maximum
prescribed plane speed. Speed includes a full-step force increment; spin is bounded by the inverse-
inertia norm times angular momentum plus a full-step torque increment. This also
covers changing angular speed during torque-free anisotropic rotation. Existing penetration, finite-state, domain and capacity checks remain
active. These motion guards prevent grossly unresolved flight; they do not detect
every possible contact that starts and ends entirely between sampled endpoints.
High-speed impact requires a separate timestep/refinement study.

The reduction shares the existing per-particle force-gather traversal, with no CPU
contact loop and no floating-point atomic accumulation. Two additional device
dispatches check the previous committed network before drift and the trial network
before the final kick. A new contact that exceeds the bound sets error bit8192.
The ordinary host advancement rollback restores the complete committed chunk,
including the estimates and friction history. Pure timestep failures never trigger
capacity-growth retries. Estimates are derived state, recomputed from geometry
and contacts on restart; checkpoint format is unchanged. Profiling through
`profile_step()` rejects this opt-in policy because that benchmark-only path lacks
transactional rollback; use ordinary `advance()` for its execution.

Storage adds 16 bytes per particle plus a 16-byte rollback copy. The conservative
memory preflight includes both. Existing default-policy runs allocate these rows
but omit the extra check dispatches. No speed benchmark is inferred from a larger
permitted step.

## Qualification and retained revisions

`examples/qualify_contact_timestep.py --output <fresh-directory>` freezes source,
shader, tests, examples, studies and case definitions before running the complete
serial Metal suite. It retains the pytest log, JUnit results, archive and a source
integrity report. Re-execution requires this project's Python environment and an
actual Metal device; there is no CPU simulation fallback.

V1 passed 299 tests with eight skips before review added the Coulomb coupling
term and dense assembly test. Its exact source snapshot and result remain in
`evidence/active-contact-timestep-v1/`; that identity does not qualify the revision.
V2 retained two failures: the dense assembly rejected its requested 8ns step;
the active-patch test supplied an analytical sphere's display-only mesh, correctly
rejected by the physical-mesh contract. V3 added a 4ns dense candidate and asserted
atomic rollback of the unchanged 8ns rejection. Its physical faceted-mesh patch
had no active nodes at the declared shallow depth, so the nonempty-contact gate
failed. V4 refined only that fixture's triangle quadrature from level 2 to level 3,
retaining the physical pose and all gates, and added independent unequal-inertia
pair-rate checks. V4 passed 302 tests/eight skips. V5 strengthens the spin guard
using an orientation-independent angular-momentum bound and corrects a reference
comment; it passes 302 tests with eight skips in 55.16s, with all 222 frozen inputs
unchanged. Skips are environment-dependent optional checks retained in the log;
no new Atlas native binary is built. All prior snapshots/logs remain.

The current frozen gates are in `tests/test_contact_timestep_policy.py`: independent
assembled spring-matrix algebra; independent flat-face elastic/viscous rates;
checkpoint-derived estimates; exact state rollback on oversized steps, excessive
spin and newly forming contact; three-grain and dense 27-grain rotating/frictional
mesh refinement; unequal-size anisotropic pair rates; and a shallow curved
active-patch comparison with the old whole-surface estimate. Refinement uses
1,2,4,8ns over 2.048us, comparing endpoint position, velocity and rotation on declared
scales. Dense8ns is required to reject and restore; its1,2,4ns candidates are
compared, while the chain includes8ns. The six-face-node cubes are synthetic
software fixtures, not converged irregular-grain quadrature. An exact matrix bound
for frozen linear contacts does not establish a nonlinear dynamic solution.
Static rate gates are 2e-5 relative; endpoint position/d and normalized velocity
limits are 1e-3, rotation limit 1e-3rad, frozen before execution.

The shallow physical faceted mesh gives an initial elastic estimate 0.997563us,
versus 7.52827ns from the whole-surface preflight, a 132.509-fold estimate ratio.
The test actually advances 32 steps at twice the legacy limit (15.0565ns); it does
not execute at 0.997563us or establish a 132-fold speedup. This fixture verifies
that only an active patch enters the estimate; it is not a quadrature-convergence
or maximum-timestep study. Values are retained as JUnit properties.

Full irregular-grain dense assembly, loading-rate, stiffness and quadrature
qualification remain open. The previous 112 normalized-stiffness trajectories
motivate this policy but are not rerun or relabelled as its qualification. The
stopped Hostun preparation is not restarted and its historical timestep remains
unchanged. No material calibration or scientific admission follows from these
software checks.
