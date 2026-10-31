# Plane traction operator and integration contract

Status: implemented fixed-normal translating planes in the Slang/Metal Engine,
including nonperiodic neighbours, source-sample collection, rigid-body forces,
reactions/work, prescribed motion, transactional retry and checkpoint state.
The independent operator probe and analytical references remain separate evidence.
This is an explicit project wall variant; no published assembly reproduction,
restitution dashpot or material qualification is claimed.

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
and the assembly implementation preserves both force and moment balance.
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

## Production laboratory integration

Set `periodic: false` and provide a `planes` list. Every plane requires `normal`
(unit inward normal), `offset` in metres, `velocity` in m/s, `normal_stiffness` and
`tangential_stiffness` in Pa/m, and nonnegative `friction`. There are no inherited
plane contact properties or automatic normal normalization. The normal stays fixed;
translation updates the offset by `dot(normal, velocity)*dt`. Tangential plane
velocity contributes to friction without changing its geometric offset.

Persistent keys are `(source_body, body_count + plane_index, source_sample)`.
Plane ordering is part of continuing history identity. Source transformations,
bounding-sphere rejection and the mesh sample tree run on the GPU. The mesh path
rejects whole sample subtrees using conservative plane/AABB support bounds; SDF
mode visits the same source samples directly. Both append to the capacity-checked
contact buffer, then reuse the committed friction hash and particle contact lists.
Plane contacts use full one-sided traction, energy and loss. Grain contacts retain
their two directed half-weighted passes. Wall spring transport uses the existing
minimal-normal rotation and mean normal spin, with zero wall angular velocity.
Rotating plane normals and inertial walls are unsupported.

Each particle gathers its wall force and common-node torque. Contact virial adds
`-force_on_body outer centre_to_node_arm` for plane contacts, with full weight;
particle/particle accounting remains unchanged. Per-plane/body rows gather opposite
reactions and common-origin moments; a GPU tree reduces those rows over bodies.
Active counts are reduced as integers, without floating point force atomics.
Per-plane outputs contain offset, velocity, reaction, origin moment, active-node
count, power and cumulative work. Fabric/coordination still count grain pairs only.

Actuator power is `dot(force_on_body, wall_velocity)`; compression work is positive.
Every step integrates old/new endpoint power by the trapezoidal rule. A prescribed
velocity change refreshes the old power using the newly prescribed velocity before
advancement. Compensated total wall work is separate from cell work, and per-wall
work has its own compensated accumulator. The laboratory proxy balance includes
both work channels. These are declared discrete work diagnostics, not an assertion
of an exact potential for all evolving contact geometries.

`kind: "wall"` stages require an explicit `wall_velocities` array, one three-vector
per plane, plus the usual steps and optional global drag. Velocities persist until
another wall stage overrides them. A hold stage preserves the current velocities;
it does not implicitly stop a moving wall. Nonperiodic affine cell rates and
periodic pressure/strain stages are rejected. No wall pressure servo is supplied.

Nonperiodic domains use clipped neighbour bins, Euclidean centre branches and no
wrapping. Their computational domain stays fixed, and centres must remain in
`[0, cell)` on each axis. Grain surfaces may extend outside it: the computational
bounds do not create contact forces. Escape rejects the entire chunk with flag
2048. Nonfinite positions fail separately. Planes and domain limits are distinct.

An optional `boundary_box` lists six distinct plane indices ordered xmin, xmax,
ymin, ymax, zmin, zmax with exact inward axis normals `+x,-x,+y,-y,+z,-z`.
Its initial and evolving gaps must be positive and its faces must stay inside the
computational domain. Invalid geometry rejects the chunk with flag 4096. These
wall gaps define measurement volume, engineering strain and nominal void ratio.
Diagnostics retain the separate `domain_cell`. Without this explicit box, volume
is the computational-domain volume and is labelled as such; it is not inferred
from arbitrary planes or particle bounds.

Wall offsets/residuals, velocities, per-body/per-wall work and reaction rows join
GPU chunk backup/rollback and capacity growth/retry. Wall checkpoints use /4 and
wall snapshots /3. Restart validates the complete plane specification, restores
state/history and re-evaluates forces at dt=0. Plane-free checkpoints remain /3;
older experiments retain their historical periodic default. Snapshot export accepts
both old and new versions. JSON histories retain plane state/reactions; CSV exposes
total wall work. No wall surface mesh is added to particle VTK exports.

## Integration evidence and executable example

`wall-boundaries-metal-v1` passes ten prospectively frozen checks, including the
32768-node sphere-cap force/energy integrals, common-origin action/reaction,
original-node wall virial, exact rotated-box source keys for oblique planes,
normal-motion work/energy, friction separation/return, restart, capacity growth,
rollback, measurement geometry and laboratory checkpoint/export semantics.
`wall-boundaries-integration-v1` passes the full serial Metal suite: 143 passed,
one skipped. Its 257-body/two-plane case exercises both reduction levels and exact
integer contact counts. Mixed grain/wall histories and complete device-buffer
memory accounting also pass. Protocols, executed source snapshots and logs remain
in their distinct evidence directories. Nonperiodic-only predecessor studies
retain their original narrower scopes.

```sh
python examples/create_fixtures.py
sand-dem run examples/wall-smoke.json --output runs/wall-first --max-chunks 2
sand-dem resume runs/wall-first/checkpoint.h5 --output runs/wall-resumed
sand-dem export runs/wall-resumed --output runs/wall-export
```

The one-sphere example places a plane for its explicit seed/geometry and runs
20 prescribed-motion steps. Changing those inputs requires reviewing initial
contact geometry. It is a software smoke case, not an equilibrated specimen.
Source-derived restitution damping and complete published-benchmark input
assignments remain separate gaps in PUBLISHED_ASSEMBLY_REFERENCE.md. Long assembly,
wall-controller, material and experimental qualification remain open.

`wall-friction-return-v1` subsequently checks the production Coulomb path against
closed partial-slip sphere-cap force, torque and return-loss integrals. Every
relative error is below the frozen 0.1% limit. For trial slip magnitude S, set
`c=mu*kn`, `h=min(delta,kt*S/c)`. Integrating returned-spring loss over the cap gives
`2*pi*R*c*(S*h^2/2-c*h^3/(3*kt))`. The imposed positive-x plane motion gives a
negative-x trial spring, positive-x force and negative-y torque. The complete
serial Metal suite at this identity passes 144 tests, one skipped. The deliberately
large first-step tangential increment is a return-map check, not a quasistatic or
large-step work-accuracy claim.

The frozen `wall-cli-smoke-v1` also passes the command sequence above, interrupted
at step 10 and completed at step 20. All five observations retain two active wall
nodes, final wall work is positive, cell work is zero, and the new snapshot version
exports to particle VTK alongside the explicit synthetic CSV/JSON bundle.

The final reviewed build (`wall-final-integration-v1`) passes 145 tests with one
skipped. It additionally rejects changed checkpoint plane normals and wall velocities
outside normalized FP32 range before mutation, and removes an unintended reduction
buffer reallocation from the profiling path. Earlier study snapshots retain their
exact executed sources and results; this final identity reruns the complete suite
and repeats the identical CLI workload without relaxing numerical gates.
