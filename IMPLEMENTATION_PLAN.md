# Implementation board

Updated 2026-10-31. Scope: a standalone Slang/Metal research solver connecting
source-traceable Sand Atlas shapes to explicitly synthetic Constitutive Atlas
comparison evidence. No changes to either adjacent atlas. This board distinguishes
implemented software from studies that still require scientific acceptance.

## Architecture and implemented sequence

1. **Metal execution foundation — implemented.** Python/SlangPy orchestration;
   explicit Metal selection, compiled-device doctor, packaged Slang kernels,
   normalized FP32 simulation units, FP64 independent reference. Fail instead of
   falling back to CPU. Record backend and shader identities.
2. **Shape pipeline — implemented for admissible solids.** Explicit SI scale,
   strict closed single-solid topology, COM/inertia, padded scalar level sets,
   stable area-weighted surface samples, versioned HDF5 and source hashes.
   Exact coordinate welding is file decoding; repairs require a separate reviewed
   asset. Synthetic sphere and original concave-prism fixtures exercise the pipeline.
3. **GPU interaction pipeline — implemented.** Periodic spatial bins, conservative
   bounding radii/Verlet skin, canonical pairs, hierarchical scan, count/emit active
   surface contacts, persistent sample-key friction hash, linked contact adjacency
   and particle force gathering without floating atomic additions. Fixed buffers
   detect overflow; advancement can restore, grow within budget and retry.
4. **Contact/integration contract — implemented as a project variant.** Two
   half-weighted surface passes, area-weighted normal traction, transported
   tangential springs and Coulomb return, equal/opposite forces at common points,
   kick–drift–kick translation and spatial angular momentum/quaternion rotation.
   Optional shared triangle BVHs now replace the field query when explicitly
   requested; exact-feature distance/static-force checks pass (docs/MESH_QUERY.md).
   Read docs/CONTACT_METHOD.md before interpreting stiffness or energy.
5. **Virtual laboratory — implemented, bounded controller case qualified.** Orthogonal
   periodic cell, affine streaming, hold/strain/isotropic/triaxial stages, explicit
   pressure filter/rate limits, diagnostic-boundary controller updates, strict
   preparation convergence/failure, restartable state and GPU chunk rollback.
6. **Measurements and offline connection — implemented.** Full contact and kinetic
   stress, strain conventions, void ratio, force imbalance, kinetic/spring-energy
   proxy, overlap, pair/contact counts, fabric, coordination, sliding fraction,
   cumulative friction/drag/work diagnostics, proxy work residual, HDF5 snapshots,
   CSV, VTK and plots. Synthetic bundle identity prevents measured-data confusion.
7. **Software/numerical checks — partial scientific coverage.** Actual Metal
   compilation, scan, GPU/FP64 forces and torques, concavity, periodic wrapping,
   friction restart, affine motion, rejected-step rollback, shape identity and
   laboratory failure are tested. Independent sphere continuum-integral refinement
   passes. These checks are not complete validation of the full research method.

## Remaining study gates, in dependency order

Public workflow requirement clarified by the user: implement the named-material
case runner and direct, explicitly synthetic Atlas handoff described in
[CASE_WORKFLOW.md](docs/CASE_WORKFLOW.md). Named-material registration, hash-checked
resolution, case overrides, dry-run and automatic exports are now implemented.
Reviewed Hostun acquisition/preparation is implemented; additional material recipes
and dedicated synthetic comparison overlays remain open. The recorded-path
adapter now uses the actual Atlas converter and browser contract with a typed
synthetic sidecar and explicit policy. This is required software
work alongside the scientific gates below; naming a material must not infer calibrated properties.

- [ ] Complete Hostun shape-library qualification. The direct archive and dimensioned
  descriptors resolve the earlier source/scale blocker; 53 valid grains are prepared
  at 32/512 with a complete identity crosswalk (docs/HOSTUN_ACQUISITION_V2.md).
  The invalid grain remains excluded as a whole, without repair. Both 32/512 and 64/2048 libraries are complete. Frozen surface-distance
  refinement passes (3.29-fold pooled p95 improvement); contact/quadrature and
  assembly convergence remain required. The source-triangle force audit fails
  the frozen 10% gate (48.39% worst at 64 cells); see
  docs/HOSTUN_CONTACT_ACCURACY.md. Current geometry is not material-qualified.
  Subdivided Gauss3 assets and conservative GPU sample hierarchies are implemented.
  The first ten-pose GPU quadrature study failed its 0.1% fixed-reference gate;
  an exhaustive per-node audit identified a proximity tie-substitution error in
  the FP64 reference. The new strict-minimum reference retains the same source
  poses, refinement levels and thresholds. Preserve the failed study;
  the separately identified strict reference and GPU requalification now pass all
  ten original poses. Worst refined integral error is 1.28981%, and same-quadrature
  GPU error is 0.01504%, below the unchanged 10% and 0.1% gates. This is bounded
  five-grain static coverage, not whole-library or assembly admission.
  The strict-minimum 2,048-node oracle and all ten corresponding GPU force/count
  checks now pass. A separately frozen whole-library source-pose/reference study
  passes for all 53 accepted grains at both prescribed sampled depths (106 cases),
  with no grain omitted. Corresponding whole-library GPU force/count checks also
  pass all 106 cases, with worst force error 0.02588% and exact active counts.
  This verifies the fixed 2,048-node computation only; full-library
  refined quadrature, dynamic contacts and assembly convergence remain required.
  Expanded fine-node comparisons exposed randomized Trimesh parity fallback;
  three strict pruning-equivalence attempts remain failed and incomplete. A direct
  solid-angle audit found batch/repeat-dependent labels at real Hostun points.
  The stochastic full-library study was stopped with its ten completed cases
  preserved. The independently frozen deterministic replacement uses bidirectional
  parity and exhaustive winding for every disagreement. All 1,024 disputed-point
  regression checks pass; the analytical integral/culling checks also pass.
  `hostun-library-quadrature-deterministic-v1` is running all 106 original poses
  at levels zero through four, with the unchanged 2% finest-change gate. Its
  completion and passing full-library GPU comparison remain required.
  Its completed shallow grain-00038 row exceeds the level-four reference gate
  (8.0767% change). The separately frozen targeted level-six study retains the
  same pose, reproduces all five overlapping levels exactly and reduces the final
  change to 0.3765%, passing the unchanged 2% gate. The whole-library parent keeps
  running, with no case removed. Full-library reference consolidation and GPU
  comparison remain required; this targeted diagnostic does not admit the L2 asset.
- [X] Freeze and run rolling/objectivity tests, separating/returning contact,
  changing normal orientation, strongly anisotropic torque-free rotation and
  capacity-growth stress tests. Test timestep refinement over useful physical time.
  Evidence: mechanics-compensated-v1 and mechanics-fast-compensated-v1, detailed in
  docs/MECHANICS_QUALIFICATION.md. These bounded engineering cases pass; Remaining FP32
  force/interpolation error and full-assembly qualification remain limitations;
  tiny state increments now pass the frozen precision study (docs/PRECISION.md).
- [ ] Freeze and run energy/work studies in static-cell, drag and affine-cell cases.
  Bounded static-cell, exact drag and contact-free affine cases pass. The frozen
  `prepared-sphere-assembly-work-v1` extends coverage to the qualified 216-sphere
  assembly under static, drag and cyclic affine conditions at three timesteps;
  all nine cases and three timestep comparisons pass. Worst proxy-balance residual
  is 0.05120%, and independently integrated work differs by at most 0.0001164% of
  initial energy. Mesh-assembly energy coverage remains open. The reported elastic energy is a proxy; work residual is diagnostic, with no
  asserted exact potential/conservation identity. Audit signs and quadrature.
- [X] Freeze controller settings independently of the final material study. Require
  pressure tolerance 1%, kinetic/elastic proxy <=1e-5, unbalanced force <=1e-3 for
  100 consecutive diagnostic samples unless a separately justified policy is
  declared before execution. Reaching step budget fails preparation. The bounded
  216-sphere case passes at 617,425 steps with all 100 samples independently audited
  (controller-v4); exact parent solver identity and earlier failures are retained.
  This does not qualify Hostun preparation.
- [ ] Prepare and run the 10,000-grain Hostun pilot with declared material parameters,
  seed, pressure, density, loading rate and timestep. Preserve failed preparations.
  Repeat timestep, shape discretization and loading-rate studies; assess quasi-static
  response and stress/strain conventions before comparing continuum results.
- [ ] Run the 50,000-grain irregular-shape performance study: sustained thermal
  operation, full memory accounting, neighbour/contact statistics and resolution
  sensitivity. Sustained 10k/50k irregular isolated-pair studies at 32/512 and
  64/2048 are complete with matched source/shader identities and memory inventory.
  Exact-mesh enclosure/compaction isolated-pair timings are also complete at
  15.20/60.32 ms per step for 10k/50k. Dense-specimen performance remains open.
  Sphere throughput is only an engineering baseline.
- [ ] Reproduce a complete published assembly benchmark and compare with independent
  experimental evidence. Calibration and validation datasets must remain distinct.
  Published wall-confined sphere benchmark source bytes are acquired with verified
  publisher checksums; all 19 numerical files are inventoried. A load file duplicates
  displacement data, and curve-column/initial-state inputs remain unresolved.
  The standalone Metal plane-traction operator now passes frozen analytical force,
  torque, energy and history checks. The nonperiodic foundation now passes the full
  131-test Metal suite (one skipped), including bounded neighbours, six-face escape
  rollback and checkpoint/export boundary identity. Production plane collection,
  prescribed motion, reactions/work, measurement-box geometry and wall-state
  checkpoint/rollback integration are now implemented. The final complete suite
  passes 145 tests with one skipped on Metal, including independent analytical
  partial-slip force/torque/loss checks and multilevel reaction reductions.
  Wall CLI interruption/resume/export also passes. Source-derived restitution
  damping and benchmark input reconstruction remain required. See docs/WALL_CONTACT.md,
  docs/CONTACT_SOURCE_AUDIT.md and docs/PUBLISHED_ASSEMBLY_REFERENCE.md.
  A later primary-source damping audit and independent oscillator/area-scaling
  checks now pass their bounded analytical gates. They demonstrate why applying
  an ordinary nodal restitution coefficient without a mass convention is
  nonconvergent. CONTACT_DAMPING_DESIGN.md declares the separate traction-viscosity
  extension to implement; no benchmark-specific mapping or dashpot is claimed.
- [ ] Complete atlas adapters and material comparison. The bounded recorded-path
  adapter checks protocol, signed stress/strain mapping, prepared origins, frozen
  approximation declarations and synthetic provenance; actual Python/browser
  request contracts pass. Qualified DEM loading, continuum execution/comparison
  and dedicated synthetic overlays remain open (docs/ATLAS_BRIDGE.md).
- [ ] Verify a second backend/device separately before advertising portability.

## Indicative effort and cost

Remaining engineering qualification: approximately 2–4 focused engineer-weeks;
accepted geometry and the first qualified material study: another 2–6 weeks depending
on source access and convergence. These are planning allowances, not completed work
or a commitment to scientific admission.

Measured short M4 Max sphere baselines: roughly 4.2 ms/step at 10k grains,
32-cell/512-sample discretization; 10 ms at 10k and 64/2048; 17 ms at 50k and 32/512.
One million steps at 10 ms costs about 2.8 hours of local runtime, before setup,
diagnostics/output and long-run thermal effects. Physical seconds depend directly on
stable timestep: at 1e-7 s, one million steps advances only 0.1 s. Dense irregular
contacts and higher resolution can be substantially more expensive. No cloud spend
is required for this Mac path. No monetary cloud estimate is claimed without a
verified backend and provider quote.
