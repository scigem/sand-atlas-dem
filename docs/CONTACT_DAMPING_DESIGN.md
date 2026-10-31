# Contact damping: source audit and explicit traction contract

Status: independent dimensional/scalar analysis, not implemented contact damping.
The current Engine has global drag and elastic/frictional contact, without a
normal restitution dashpot. A restitution number must not silently select an
unverified nodal mass or coefficient conversion.

## Bounded primary source

Ziran Zhou, *Predictive Modeling of Architected Solids across Scales* (Caltech
Ph.D. thesis, 2025), [DOI 10.7907/cn82-z804](https://doi.org/10.7907/cn82-z804).
The [author PDF](https://thesis.caltech.edu/17018/1/Zhou_Ziran_2025.pdf) is retained
privately with receipt in `evidence/contact-damping-source-audit-v1/receipt.json`.
SHA-256: `fe19141cac9df2a26291277fabbcd1f61de113a093f105fa8e5bcc193a4c3839`.
PDF title/copyright pages and PDF page 29 (printed 15) were visually inspected.
Copyright reserves all rights. This is the thesis version, not an exact-copy audit
of the related 2023 journal article or of Feldfogel's implementation.

Equations 3.2-3.5 describe nodal spring/dashpot normal force, a restitution-based
coefficient, incremental shear and Coulomb limiting. The text identifies m as the
colliding object's mass. The printed denominator puts a superscript two on the
restitution symbol after the logarithm; its intended parenthesization is ambiguous.
No silent correction or transfer of its force/normal sign assignment is made.
These equations do not specify the later area-weighted benchmark's mass allocation.

## Independently derived ordinary oscillator

For the scalar model `m*x'' + gamma*x' + K*x = 0`, with `x(0)=0`, `x'(0)=v0`,
`0 <= zeta=gamma/(2*sqrt(m*K)) < 1`, the next zero of x occurs at
`tc=pi/(sqrt(K/m)*sqrt(1-zeta^2))`. At that instant:

`e=-x'(tc)/v0=exp(-pi*zeta/sqrt(1-zeta^2))`.

Solving gives `zeta=-ln(e)/sqrt(pi^2+(ln(e))^2)` and
`gamma=2*zeta*sqrt(m*K)`, with `K` in N/m and gamma in N*s/m.
This derivation independently establishes the squared-log interpretation for that
particular unclipped linear oscillator. It does not establish an author's intent,
a multi-contact reduced mass, or a clipped/nonlinear contact restitution law.

`restitution-scaling-audit-v1` compares this solution with an independent FP64
`solve_ivp` integration at e=0.1, 0.5, 0.9 and 1. Maximum endpoint error is
1.16e-12, below the frozen 1e-10 gate. Eight independent quadrature/parameter
checks also pass. This is an analytical reference calculation, without DEM dynamics.

## Why a nodal substitution is insufficient

Let a fixed contact patch of area A have N equal tributary nodes, each with
`K_i=kn_star*A/N`. If every node uses the entire body mass m, the summed
`gamma_i=2*zeta*sqrt(m*K_i)` grows as sqrt(N). The audit's refinement sequence
N=1,4,16,64,256,1024 increases damping by 1,2,4,8,16,32. This is a diagnosed
nonconvergent alternative, not a passing solver formulation.

Allocating `m_i=m*A_i/A_total` instead makes gamma_i proportional to area and
removes that counting divergence. But this is an additional mass-distribution
assumption, not acquired benchmark evidence. For a translational patch occupying
fraction f of A_total, its effective ratio is `zeta*sqrt(f)`, so the nominal
restitution is recovered only for the full patch in that restricted oscillator.
For target e=0.5, f=0.01,0.1,1 give effective linear restitutions approximately
0.93454,0.80691,0.5. Rotational mobility and a second finite body add further choices.

Putting kn_star directly into an ordinary `sqrt(m*K)` formula also has the wrong
units. Traction stiffness is N/m^3; normal traction viscosity must be N*s/m^3,
whereas an ordinary nodal dashpot is N*s/m. Its area factor and mass convention
must be derived, not supplied by a restitution label.

## Explicit project extension to implement

Use an optional, explicitly supplied normal traction viscosity `cn_star` in
N*s/m^3. Zero retains the existing undamped method. For a source node of area A,
penetration d>0, inward normal n and source/target relative normal velocity vn:

- Elastic normal traction is `kn_star*d`.
- Total repulsive traction is `max(0,kn_star*d-cn_star*vn)`.
- Apply the existing directed pass weight (one half for grain passes, one for planes).
- Coulomb limiting uses the actual repulsive normal force.
- Retain elastic spring energy as a proxy; compute normal dissipative power from
  `weight*A*(kn_star*d-total_traction)*vn`, which is nonnegative for this clamp.
- Keep normal contact dissipation separate from discrete Coulomb loss, global drag,
  affine cell work and imposed wall work. Integrate power consistently in the
  kick/drift/force/kick stages, with compensated cumulative state.

This specifies a dissipative project contact law; it does not promise any fixed
collision restitution. Declare the viscosity for grain contacts and each plane
explicitly. Do not calibrate it silently from e, density or a guessed nodal mass.
For a translating spherical cap with inward speed u>=0, before unloading/clipping,
`A_contact=2*pi*R*d`, `F_el=pi*kn_star*R*d^2`,
`F_damp=2*pi*cn_star*R*d*u`, and `P_loss=2*pi*cn_star*R*d*u^2`.
The local tangent damping ratio is proportional to sqrt(d), even for fixed cn_star.
Scalar surface quadrature independently verifies these integrals.

Implementation must extend conservative viscous timestep bounds, GPU force/history
and reduction paths, checkpoint state, rollback/growth, versioned outputs and work
balance. Freeze compressive/unloading/clipped nodal tests, cap-integral convergence,
constant-area analytical oscillators, timestep refinement, action/reaction,
objectivity, restart and zero-viscosity regression before executing studies.
A benchmark-specific restitution mapping still needs source evidence or a separately
identified and justified calibration study, followed by independent validation.
