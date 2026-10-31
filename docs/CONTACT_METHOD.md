# Contact and integration contract — implementation v0.1

## Source and project identity

The baseline is Feldfogel, Karapiperis, Andrade and Kammer,
*A discretization-convergent level-set-discrete-element-method using a continuum-based
contact formulation*, IJNME 125(5), e7400 (2024; online 2023-11-20).
Persistent source: https://doi.org/10.1002/nme.7400 . Readable source consulted:
https://onlinelibrary.wiley.com/doi/full/10.1002/nme.7400 on 2026-10-31.
Equations (4)–(8), sections 2.3–2.4, specify normal area-weighted traction; the adjacent
text specifies the analogous area factor for tangential stiffness. Figures 1 and 3
provide the two-sphere refinement setting. The complete original LS-DEM source and
all published numerical benchmark inputs have not been independently reproduced.

The paper's normal nodal force is `kn_star * penetration * normal * tributary_area`.
`kn_star` and `kt_star` have units Pa/m = N/m³; they are not ordinary pair stiffness
in N/m and are not automatically Young's moduli.

The implementation is **an explicit project numerical variant**: both directed
surface-to-field passes receive half weight, using area-stratified stable samples,
trilinear SDF interpolation or explicitly selected original-triangle distance,
an explicit spring/Coulomb update and the integrator below. This is not a
byte-for-byte reproduction of an author solver, an assertion
of exact elastic energy conservation, or Hertz–Mindlin contact.

## Geometry and forces

- Body frame is at centre of mass; orientation quaternion is xyzw, body to spatial.
- Signed distance is negative inside. In SDF mode the normal is the normalized
  gradient of the trilinearly interpolated scalar field. Mesh mode uses the signed
  closest-point distance gradient, with feature pseudonormals determining sign;
  see MESH_QUERY.md. Singular gradients fail the step.
- For each directed source node penetrating its target by d, area A:
  `Fn = 0.5 * kn_star * A * d * n`.
- Evaluate the two directed passes, preserving disconnected active regions.
- Source and target receive opposite forces at the same source-node position;
  torques use the respective centre-to-node arms. This gives pair force/moment
  balance even when sampled normal forces are not central.
- Surface samples use deterministic scrambled Sobol face-area sampling, equal
  tributary area `total_surface_area / sample_count`. Original synthetic sphere
  fixtures use a deterministic Fibonacci sphere distribution.
- Separate face-quadrature assets use source-face centroids or three-point Gaussian
  nodes with their explicit triangle-area weights. Asset identity and sample keys
  change; these assets cannot silently replace a continuing friction history.
- Geometric distance predicates run on order-one normalized geometry before
  converting distances to metres. The original unnormalized implementation failed
  a concave-grain comparison; `evidence/verification-v2.log` retains that result.

## Friction and state

Persistent key = `(source_particle, target_particle, source_sample)`; two directed
passes have separate keys. A compact hash references the prior committed active
records. Normal frame transport uses minimal rotation plus common-body normal spin.
A near-antiparallel normal transition resets the local spring and is an explicit
nonsmooth event, not an energy-conserving update.

At a contact update: transport committed slip, add `tangential_relative_velocity * dt`,
compute `Ft = -0.5 * kt_star * A * slip`, project onto the Coulomb disk
`|Ft| <= mu * |Fn|`, then make the elastic spring consistent with the returned force.
Dissipation from return mapping is recorded. Slip is deleted on separation; neighbour
rebuild and periodic wrapping do not reset a continuing contact. There is no added
rolling resistance or cohesion. Optional explicit `normal_viscosity` adds a
repulsive clamped area-weighted normal traction dashpot; see
CONTACT_DAMPING_DESIGN.md for the law, units, force timing, separate loss channel
and bounded verification. It does not infer a restitution coefficient.

## Periodic cell and motion

The cell is diagonal and orthogonal. Minimum image is unique only while maximum
pair cutoff is less than half the shortest cell dimension. Reject violations.
Use a linked spatial grid, conservative radii and a Verlet skin. Rebuild when the
per-particle displacement plus a conservative cell-deformation bound exceeds half
that skin. Bin ranges widen as the cell contracts; repeated bins are not visited.

Velocity is peculiar velocity. Contact relative velocity includes angular velocity
and the periodic affine streaming difference `-L * centre_branch`. The deformation
split maps positions with `exp(L*dt)` and peculiar velocities with `exp(-L*dt)` around
translation drift. This is a homogeneous diagonal deformation algorithm; arbitrary
sheared-cell dynamics and physical poromechanics are outside this version.

Motion uses kick–drift–kick; angular momentum is spatial, angular velocity comes
from rotated inverse inertia, and orientation uses midpoint quaternion rotation.
Drag exponentially damps translational/angular velocities during declared stages.
The timestep is fixed and checked against a conservative whole-surface mass/stiffness
and inertia/stiffness estimate. That estimate is not a proof of stability.

Normalization: length L0 = median library equivalent diameter; mass M0 = rho*L0³;
time T0 = sqrt(rho*L0/kn_star); force F0 = kn_star*L0³; stress P0 = kn_star*L0.
GPU data is FP32. Source properties and the independent reference are FP64.
Tiny state changes use compensated accumulation with explicit rounding boundaries;
checkpoint /3 includes these residuals. See PRECISION.md for the analytical checks
and retained compiler/roundoff failures.

## Trial, commit and failure

An advancement chunk is a transaction. Position, velocities, orientation, angular
momentum, cell, forces, neighbour state and committed contact history are backed up
on the GPU. Error flags prevent subsequent integration; the entire chunk rolls back.
Capacity errors can grow buffers within the working-memory budget and retry from
that committed state, with an explicit retry log. Invalid gradients, nonfinite values,
periodic ambiguity and excessive overlap remain failures. No force is truncated.
Initial allocation overflow currently fails preflight/initialisation explicitly;
automatic growth applies to advancement after a committed initial state.

All state needed for continuing friction is included in a versioned checkpoint.
On restart, validate geometry identity, restore histories, rebuild their hash and
re-evaluate contacts at dt=0. Cross-device bitwise equivalence is not promised.

## Measurements and limitations

Contact stress = `-sum(force_on_source outer minimum_image_centre_branch)/cell_volume`,
counting every directed nodal interaction once. Add peculiar kinetic stress separately.
Retain the nonsymmetric tensor and report its antisymmetry; symmetrize only for p/q.
Nominal void ratio uses the sum of rigid-grain volumes and ignores overlap volume.
`elastic_energy_proxy` is the sum of nodal normal/tangential spring energies. It is
not established as a conserved potential of the moving sampled contact geometry.

Bounded rolling/objectivity, high-spin refinement and sphere-controller cases pass.
Remaining release gates include mesh-contact quadrature, extended assembly energy/work
studies, published whole-assembly reproduction, calibrated preparation settings,
a complete 10,000-grain material study and physical experimental comparison.

Cumulative work and dissipation are reported in joules: Coulomb return loss,
exponential drag loss, and trapezoidal compression-positive cell-stress power.
Fabric is the equal-weight dyadic average of centre-branch unit normals over
active grain pairs; coordination is twice active grain pairs divided by all grains.
The work residual subtracts kinetic and elastic-proxy changes from cell work minus
losses. Its quadrature and proxy-potential interpretation remain qualification
work; the exact no-contact exponential drag case has an independent regression check.

Fixed-normal translating planes are an explicit additional project boundary
variant; see WALL_CONTACT.md. They require nonperiodic neighbours, full one-sided
area weighting, common-node reactions and separately integrated wall work.
Optional orthogonal wall gaps provide the declared specimen measurement volume;
the computational domain otherwise stays distinct and fixed. Plane motion,
velocity schedules and compensated work participate in rollback and checkpoint /4.
The plane-free periodic contract and checkpoint /3 remain supported. The current
full Metal suite passes 145 tests with one skipped, including independent bounded
plane force/torque/return-loss and work checks. No restitution dashpot or published
wall-confined assembly reproduction is inferred from those results.
