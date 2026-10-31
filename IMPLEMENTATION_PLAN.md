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

Current steering: the user stopped the ongoing Hostun preparation after its
million-step checkpoint to study individual-contact timestep limits. Completed
96-case analytical and32-case coordinate-translation Metal sweeps are retained in
[CONTACT_TIMESTEP_LIMITS.md](docs/CONTACT_TIMESTEP_LIMITS.md). They expose excessive
conservatism in the existing preflight for these fixtures and a coordinate-error
floor. Production policy remains unchanged. Irregular pair contacts, rotation,
friction and loading-rate/refinement remain required before a new preparation.
The subsequent [local-plane precision fix](docs/LOCAL_FRAME_CONTACTS.md) passes
12 translated static cases, all 128 original/translated dynamic repetitions and
the complete 286-pass/eight-skip Metal-enabled suite. It reduces coordinate
sensitivity while retaining the original failures and production timestep policy.
Earlier running-preparation statements below describe historical progress.

Public workflow requirement clarified by the user: implement the named-material
case runner and direct, explicitly synthetic Atlas handoff described in
[CASE_WORKFLOW.md](docs/CASE_WORKFLOW.md). Named-material registration, hash-checked
resolution, case overrides, dry-run and automatic exports are now implemented.
Reviewed Hostun acquisition/preparation is implemented; additional material recipes
and dedicated Atlas browser comparison overlays remain open. Ottawa 1 µm source intake now retains all 175 meshes, the 244-row descriptor table, invalid members and unresolved identity/scale correspondence (docs/OTTAWA_SOURCE_INTAKE.md); no recipe is admitted. The recorded-path
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
  `hostun-library-quadrature-deterministic-v1` completed all 106 original poses
  at levels zero through four, but failed its final broad source-integrity gate
  after separate visualization commit ff7a766 changed cases/CLI/simulation.
  All numerical reference dependencies remained unchanged; the failed gate and
  all rows are retained in hostun-library-quadrature-terminal-audit-v1.
  Grain 54 deeper also exceeds the unchanged 2% finest-change gate (2.6276%).
  The separately frozen grain-54 level-six diagnostic passes from an immutable
  source copy (0.29146% final change, exact overlap-level agreement).
  A fresh all-106-pose adaptive reference rerun now uses an immutable source copy
  and extends failing level-four poses through level six at the same 2% gate
  (hostun-library-quadrature-adaptive-v2); its earlier script-hash preflight
  rejection is retained. Full-library reference consolidation and GPU comparison remain required.
  Its completed shallow grain-00038 row exceeds the level-four reference gate
  (8.0767% change). The separately frozen targeted level-six study retains the
  same pose, reproduces all five overlapping levels exactly and reduces the final
  change to 0.3765%, passing the unchanged 2% gate. The terminal parent retains every original case. Full-library reference consolidation and GPU
  comparison remain required; this targeted diagnostic does not admit the L2 asset.
  Fixed per-shape quadrature selection/preparation and mixed-level GPU study
  support are now implemented. Selection requires every pose and a passing,
  source-stable archived reference; it cannot use a level as its own reference.
  Physical geometry, mass and inertia stay unchanged. Balanced shape sampling
  supports complete source inventories in subsequent assemblies. The final
  combined suite passes 201 tests on Metal with three optional rendering skips.
  The final analytical-workflow integration passes 202 tests on Metal, including
  actual Atlas consumers, with three optional rendering skips and unchanged frozen
  sources. The separate analytical box workflow now passes six independent reference and
  twelve actual-Metal coarse/fine cases with selected levels 4, 4 and 2. Worst
  integral error falls from 20.9604% to 4.8583%, preserving the existing gates and
  both earlier failed studies. This verifies mixed workflow execution, not Hostun.
  A source-frozen accepted-reference resolver now prepares the candidate and full
  212-case coarse/adaptive GPU protocol with the unchanged fixed-node baseline.
  It freezes contact capacity from selected CPU counts, instead of the insufficient
  old 8,192-node provisional allocation. Twelve analytical-fixture Metal cases and
  rejection of the actual pending Hostun reference pass; no Hostun gate is closed.
  The adaptive Hostun asset, all-106 mixed GPU comparison and assembly studies
  remain open; see docs/ADAPTIVE_QUADRATURE.md.
  Latest reference consolidation passes all 106 original cases; the additional
  deeper grain-38 level-six reference permits independent fixed level-four
  production selection at both depths. Worst final reference change is 1.8014%;
  selected levels are 51 grains at two, one at three and one at four (worst selected
  error 2.0861%). Full 212-case adaptive GPU v3 passes unchanged gates and source hashes. Worst
  integral error falls from 100% to 2.0751%; same-level GPU error is at most
  0.021873%. This closes the sampled-pose static quadrature gate. Earlier preflight,
  archive-consolidation and legacy provenance failures remain retained.
  A 424-grain/all-53-shape preparation protocol and independent geometry intake
  are frozen; the qualification-gated immutable preparation is now running, with
  an independent checkpoint/diagnostic audit required on terminal success.
  No Hostun assembly acceptance is inferred before completion.
  A subsequent same-pose level-eight challenge for grain 54 exposes nonmonotone
  coarse error: the old selected level three exceeds its 5% preprocessing target.
  Both extended references pass unchanged gates with exact overlaps. Retained GPU
  observations still pass the original 10% integral gate against stronger references
  (worst 5.04564%). A physically identical revised level-five CPU candidate is
  prepared, with GPU v4 qualification queued. The running preparation remains
  unchanged and bounded; no material gate is closed. Higher-level preprocessing
  is explicit opt-in and CPU-tested; actual higher-level Metal remains pending.
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
  The dependent all-53-shape/424-particle mesh-work workload is prospectively
  frozen at2,1,0.5ns for all nine original static/drag/cyclic cases. Its resolver
  binds accepted preparation/audit/shape hashes and executes immutable sources.
  Eight CPU contract checks and pending-preparation rejection pass; actual mesh
  work/timestep execution awaits preparation. No energy gate is marked complete.
- [X] Freeze controller settings independently of the final material study. Require
  pressure tolerance 1%, kinetic/elastic proxy <=1e-5, unbalanced force <=1e-3 for
  100 consecutive diagnostic samples unless a separately justified policy is
  declared before execution. Reaching step budget fails preparation. The bounded
  216-sphere case passes at 617,425 steps with all 100 samples independently audited
  (controller-v4); exact parent solver identity and earlier failures are retained.
  This does not qualify Hostun preparation.
- [ ] Prepare and run the 10,000-grain Hostun pilot with declared material parameters,
  seed, pressure, density, loading rate and timestep. Preserve failed preparations.
  Metadata-only resource intake now bounds the revised candidate at 12.615–12.865
  GiB for 10k and 52.966–53.216 GiB for 50k at current contact capacities, above
  the 12 GiB configured budget. Larger-case capacities and total peak memory need
  independent measurement/review; no automatic budget increase or capacity reduction.
  Repeat timestep, shape discretization and loading-rate studies; assess quasi-static
  response and stress/strain conventions before comparing continuum results.
- [ ] Run the 50,000-grain irregular-shape performance study: sustained thermal
  operation, full memory accounting, neighbour/contact statistics and resolution
  sensitivity. Sustained 10k/50k irregular isolated-pair studies at 32/512 and
  64/2048 are complete with matched source/shader identities and memory inventory.
  Exact-mesh enclosure/compaction isolated-pair timings are also complete at
  15.20/60.32 ms per step for 10k/50k. Dense-specimen performance remains open.
  Sphere throughput is only an engineering baseline.
  Symbolic execution of the current allocation declarations now establishes a
  41.858 GiB contact-storage subtotal at 50k with unchanged 4096-contact capacity,
  exceeding installed 36 GiB before geometry. Active-prefix rollback allocation
  is now implemented and CPU-verified; actual-device qualification and measured
  dense savings remain required. Further storage work may be needed; geometry-only changes
  cannot close this resource gap. See docs/CONTACT_STORAGE.md.
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
  nonconvergent. The separate explicit traction-viscosity extension is now implemented;
  normal-viscosity-integration-v1 passes all 166 tests on Metal. After merging
  existing visualization work, the combined suite passes 174 tests with three
  optional rendering skips. Viscous CLI checkpoint/5 resume and exports pass. The independent
  constant-area oscillator v2 passes at three timesteps (4.27-fold phase refinement;
  0.3327% finest cumulative-loss error), preserving v1 preflight rejection.
  No benchmark-specific restitution mapping is claimed; see CONTACT_DAMPING_DESIGN.md.
- [ ] Complete atlas adapters and material comparison. The bounded recorded-path
  adapter checks protocol, signed stress/strain mapping, prepared origins, frozen
  approximation declarations and synthetic provenance; actual Python/browser
  request contracts pass. Actual native continuum replay also passes five analytical
  fixture observations, including reversal/duplicate ordering, against independent
  Hooke stress/strain and void-evolution checks (atlas-native-replay-v2). This is
  bounded software coverage, not a qualified DEM response comparison.
  The standalone native comparison command now validates the generated catalogue,
  binary, explicit/legacy supported stress contract and complete ordered observations,
  and writes synthetic JSON plus optional SVG/PNG overlays. The five-observation
  analytical fixture passes; dedicated Atlas browser overlay integration remains open.
  Actual v0.3 barodesy-sand replay also completes all five original fixture
  observations (worst normalized control error 3.1484e-11, limit 1e-10).
  Final serial actual-Metal/native Atlas regression passes 225 tests with three
  graphical rendering skips; all frozen repository and external inputs are unchanged.
  Qualified DEM loading, continuum execution/comparison
  and dedicated Atlas browser overlays remain open (docs/ATLAS_BRIDGE.md).
  A read-only intake now binds all ten existing curated Hostun triaxial records
  and prior source-conflict reviews. State/strain and physical batch correspondence
  remain unresolved; no normalization, inferred preparation, fit or calibration/
  validation split is introduced. See docs/HOSTUN_EXPERIMENTAL_ADMISSION.md.
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

The later 53.6-million-node Hostun mesh workload has substantially different
cost: completed dilute post-contact intervals average 11.720 ms/step for 424
grains. The frozen rate cap requires approximately 6.624 million additional
steps from the 500k checkpoint to the nominal 0.45 density floor. Holding the
observed cost fixed gives about 22 wall hours for compression alone, not a
completion ETA or dense-performance claim. See docs/HOSTUN_RUNTIME_COST.md;
subsystem profiling and requalified optimization remain required.
