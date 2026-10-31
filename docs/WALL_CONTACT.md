# Plane traction operator and integration contract

Status: an independently verified Slang/Metal force operator. The laboratory does
not yet advance wall-confined specimens. `shaders/plane.slang` defines the reusable
response and an isolated probe; `plane_contact_study.py` exercises it on the actual
Metal device. It is not connected to the current periodic Engine contact collector.

## Explicit project extension

For a unit normal n pointing into the admissible region, plane offset b and source
point x, set phi = n dot x - b. A point is active only if phi < 0; d = -phi.
The normal force is `kn_star * A * d * n`. This is full one-sided area weighting,
derived by integrating the declared plane traction. It is distinct from the
particle/particle implementation's two directed half-weight passes. No equivalence
to the published benchmark's wall implementation is asserted.

Tangential velocity is the relative node/wall velocity projected into the plane.
Advance the continuing spring by that velocity times dt, then apply the pointwise
Coulomb limit `|Ft| <= mu * Fn`. The returned spring is consistent with the clipped
force. Inactive points clear the spring; returning points start with no history.
The probe's active flag belongs to that same persistent source-node/plane key.
Normals are fixed during a history update; rotating-wall transport is unsupported.

Force acts at the source node. Reaction is opposite at the same common point,
so a future assembly implementation must preserve both force and moment balance.
The probe reports torque about its origin. Its inputs are normalized world points,
positive source areas, relative velocities and explicitly active old springs.
It does not transform particle geometry or advance particles on the CPU.

## Independent analytical reference

For a radius R sphere centred at zero and a +z inward plane at b = -R + delta,
0 <= delta <= 2R, the sphere area element is `dA = 2*pi*R*dz`.
Consequently `Fn = pi*kn_star*R*delta^2`, active area is `2*pi*R*delta`, and normal
spring energy is `pi*kn_star*R*delta^3/3`. Integrating Fn over imposed penetration
gives that energy for this fixed-normal normal-loading case.

For a uniform trial spring s along +x, define a = kt_star*|s|, c = mu*kn_star and
h = min(delta, a/c), with h = delta when c = 0. Pointwise Coulomb return gives
`J0 = c*h^2/2 + a*(delta-h)` and
`J1 = c*h^3/3 + a*(delta^2-h^2)/2`.
Then `Fx = -sign(s)*2*pi*R*J0` and
`Ty = sign(s)*2*pi*R*((R-delta)*J0+J1)`. Other resultant components vanish.
These formulas use source-node moment arms, consistently with the operator.
Six independent scalar surface-quadrature/work checks verify the closed forms.
This identity is not a potential-energy assertion for arbitrary moving contacts.

## Frozen verification

`plane-traction-operator-metal-v1` covers 18 sphere-cap observations: three node
counts, three penetrations and normal-only/partial-slip loading. Every coarse
observation is retained. At 32,768 nodes the frozen force, torque and normal-energy
limits are 0.1%, on their declared analytical scales. Nodal continuation, separation,
return, Coulomb loss and rigid-frame covariance use a separate 1e-5 relative limit.
All pass. Worst fine force and torque errors are 4.55e-7 and 6.43e-6 relative.
The default sandbox exposed no Metal adapter; that failed initialization, executed
source and unchanged policy are retained as `plane-traction-operator-sandbox-v1`.
The authorized host execution has a separate identity and no relaxed gate.

## Remaining laboratory integration

The nonperiodic foundation is now implemented: `periodic: false` selects a fixed
computational domain, with clipped neighbour-bin traversal and Euclidean centre
branches. Opposite faces do not interact. The domain may be smaller than twice
the pair cutoff; the periodic minimum-image restriction does not apply. Particle
centres must remain in `[0, cell)` on each axis. Grain surfaces may extend outside
that envelope: these bounds do not generate physical wall forces. Escape rejects
the entire advancement chunk with flag 2048 and restores its committed state.
Nonfinite positions fail separately. Nonperiodic cases currently allow hold stages
only and reject nonzero affine rates before mutation.

Checkpoint /3 already stores the complete experiment, including the new explicit
boolean, so no extra state-array version is needed for this foundation. Older
experiments retain the historical periodic default. Checkpoints and synthetic
exports retain the boundary setting; manifests explicitly label nonperiodic
measurement volume as the computational domain. That volume is not an established
wall-confined specimen volume.

`nonperiodic-domain-v1` passes all eleven prospective checks. The expanded
`nonperiodic-domain-integration-v1` passes the complete serial Metal suite:
131 passed, one skipped. This covers exact unique neighbour sets including a
single-bin domain, six-face transactional escape, analytical free drift/rebuild,
state restore, SDF/mesh interior contacts and continuing friction, and HDF5
checkpoint/export semantics. Protocols, source snapshots and logs are retained.
These checks do not establish plane-contact integration or assembly acceptance.

GPU source transformations and conservative sample-hierarchy plane rejection must
feed capacity-checked collection; an O(all nodes times all particles) host loop is
not an integration path. Persistent keys must distinguish planes from grain targets.
Force/torque gathering, wall reactions, stress, imposed-wall work and dissipation
need consistent common-point accounting. Wall offsets, prescribed velocities and
history belong to trial/commit/rollback, growth/retry and versioned checkpoints.
Freeze dynamic wall motion, restart, overflow, work and assembly acceptance cases
before executing them. The existing periodic case/export semantics must remain
explicit. Restitution damping and complete published-benchmark input assignments
remain separate source/integration gaps in PUBLISHED_ASSEMBLY_REFERENCE.md.
