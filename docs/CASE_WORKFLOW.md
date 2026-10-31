# Named-material case workflow

The named-material runner and reviewed Hostun source acquisition are implemented.
The recorded-path Atlas bridge is implemented with an explicit synthetic sidecar.
Source recipes for other materials, named numerical profiles and dedicated Atlas
browser overlays remain open; standalone native comparisons are available. The existing
`run`, `resume`, geometry and export commands remain available as lower-level tools.

## User interface

Hostun can be acquired/prepared once, or prepared assets can be registered manually:

```sh
sand-dem material prepare hostun --workers 4
# Alternatively, register the separately prepared asset:
# sand-dem material register cases/hostun-material.json
sand-dem material list
sand-dem case run hostun --case cases/drained-triaxial.json --dry-run
sand-dem case run hostun --case cases/drained-triaxial.json --output runs/hostun-tc
# Sweep overrides (SI):
sand-dem case run hostun --case cases/drained-triaxial.json \
  --pressure 20000 --rate 0.5 --stop-strain 0.05 --particles 10000 --seed 7 \
  --output runs/hostun-tc-20kpa
```

Registration uses a `sand-dem-material/1` receipt containing a canonical ID, aliases,
prepared-library path/SHA-256, source receipt and coordinate-scale evidence. Libraries
are referenced in place and checked at registration and resolution. The default
registry is `~/.cache/sand-atlas-dem/materials`; `--cache PATH` selects another.
Duplicate registrations and ambiguous aliases fail. Canonical IDs take precedence.
The included manual Hostun receipt refers to a separately prepared private asset.
`material prepare hostun` instead downloads the reviewed source archive/descriptors,
checks pinned SHA-256 and member identities, rebuilds the fixed scale crosswalk,
and prepares/registers the library. First-use `case run` automatically does this
for an unregistered Hostun material; dry-run never acquires sources or starts Metal.
Other materials require a reviewed source/scale recipe or explicit manual preparation
and registration. Neither path establishes scientific qualification.

The case declares particle count, seed, density, contact stiffness and friction,
preparation pressure/history, loading rate and stopping condition. Numerical
settings may come from a named, versioned profile, with explicit overrides. Every
resolved value goes into the output manifest. A material name supplies grain
geometry and source identity; it does not supply calibrated mechanical properties.

Optional experiment `normal_viscosity` is an explicit grain-contact traction
viscosity in N·s/m³, default zero; each plane has its independent value. A restitution
number is not substituted for this coefficient. Resolved inputs retain these
values, and output histories include normal contact power and cumulative loss
separately from friction and global drag. See CONTACT_DAMPING_DESIGN.md.

`shape_sampling: "balanced"` includes every registered shape with counts differing
by at most one and requires at least one particle per shape. Default `"random"`
retains the original seeded behavior. This is numerical sampling, not an inferred
material size distribution. See ADAPTIVE_QUADRATURE.md for fixed per-grain surface
resolution and its separate reference/GPU qualification gates.

`sand-dem-case/1` case files contain an `experiment` object without a geometry path.
Density, both contact stiffnesses, friction, timestep and stages must be explicit.
`cases/drained-triaxial.json` is a provisional research template, not a qualified
Hostun preparation recipe; `cases/mesh-hold-smoke.json` is a two-step execution check.

Reusable case files are the primary reproducible interface. Equivalent command
line overrides should cover common pressure, rate, strain, particle-count and seed
sweeps. A dry-run command resolves all settings and reports the step/physical-time
budgets, asset size and configured memory limit without starting Metal. Case-specific
wall time is explicitly unknown; the memory limit is not an allocation estimate.
Low-level geometry and solver settings remain available.

## Material resolution

Resolve a unique Sand Atlas dataset identity from its canonical ID or an
unambiguous alias. Record metadata and source hashes. Reuse a verified prepared
asset from a local cache, or acquire and prepare its source geometry once. Cache
identity includes original bytes, scale evidence, excluded solids, preprocessing
version and numerical representation. Ambiguous names and unsupported scale
evidence must produce actionable errors rather than inferred defaults.

The Hostun acquisition and its excluded grain already have a documented source/
descriptor crosswalk. Other materials require their own scale/topology checks.
Cached geometry qualification and calibrated parameter provenance are separate.

## Acquisition and cache recovery

The default acquisition profile is 64-cell/2,048-sample geometry, seed zero, four
CPU workers. `material prepare` accepts `--resolution`, `--samples`, `--workers`
and `--source-directory` (reuse pinned archive/CSV bytes in that directory). The
existing fine Hostun preparation recorded about 2.1 aggregate CPU-hours; parallel
wall time depends on CPU contention. The mesh ZIP is about 24.4 MiB. There is no
GPU execution during acquisition; geometry preprocessing runs on the CPU.

Original bytes and prepared assets stay under `<cache>/private/hostun-0000005/`.
The dataset directory is created with mode 0700. The preparation identity covers
source recipe, numerical settings, Python/dependency versions and preprocessing
code hashes. Completed grain assets have separate SHA-256 receipts and can be
reused after interruption. Changed inputs select another profile directory;
registration with an existing incompatible profile fails and requests a different
`--cache`. Existing manually registered libraries remain usable by `case run`.

An exclusive `preparation.lock` records the writer PID and profile identity.
Ordinary exceptions remove it and retain a failed attempt report. After abrupt
termination, verify that the recorded process is terminal before removing that
specific stale lock and repeating the same command. Never remove an active lock.
Each attempt has a distinct report; failed source downloads leave no final source
file. Corrupt cached sources or prepared grains fail explicitly without overwrite.
A changed published archive is not automatically admitted under a new hash.

## Execution

Resolve material plus case into the existing explicit `Experiment`. Preserve
preparation convergence gates, transactions, capacity retries and checkpoints.
An unsuccessful preparation produces a failed run with diagnostics, not a valid
loading curve. Resume uses the resolved case and original asset identities.

The implemented runner writes `resolved-case.json`, `result.json`, a `run/`
directory and automatic `exports/history.csv` plus `exports/atlas-comparison.json`.
The comparison bundle links material/case identities and declares run status and
`scientific_admission: false`. Initialization exceptions retain resolved inputs and
a failed result; a solver failure preserves its available history and exports.
Plots and VTK remain available through the lower-level `export` command.

Automatically export a completed case. Preserve failed and partial histories with
their status, without presenting them as completed virtual tests. Preparation
history and loading observations remain separately identified. Loading-stage
strain and time origins use the actual prepared state.

## Constitutive Atlas handoff

Deliver an explicitly synthetic virtual-test dataset for comparison with
constitutive-model predictions. The current `sand-dem-atlas-comparison/1` bundle
retains raw DEM channels but is not yet directly accepted by Atlas's current
measurement importer. The [Atlas bridge](ATLAS_BRIDGE.md) instead uses the existing recorded-path
converter, with a separate typed virtual-test sidecar and private legacy conversion
input explicitly identified as synthetic. It does not use the measurement importer.

The handoff must retain material/geometry identity, parameters, preparation,
case and solver hashes, run status and qualification status. It needs explicit SI
units, compression-positive signs, tensor frame and strain measures. For a drained
triaxial projection, map DEM axial z to Atlas axial x and compute signed axial-minus-
mean-radial stress from the stress tensor; DEM's nonnegative J2 q is a different
quantity. Preserve the full tensor and radial asymmetry alongside any projected
curve. Finite-volume strain and engineering/logarithmic strain remain distinguished.

The target initial Atlas handoff is a dry drained-triaxial virtual experiment with
stress, axial/radial/volume strain, void ratio and physical time. Isotropic and
general strain paths need their own admitted protocol mappings. Effective stress
interpretation, axisymmetric projection and continuum kinematic compatibility
must be declared. Successful import does not establish material calibration or
experimental validation.

The separate `sand-dem atlas compare` command executes a prepared handoff through
an existing matching native Atlas solver/catalogue and writes ordered synthetic
response comparisons, with optional SVG/PNG overlays. See ATLAS_BRIDGE.md. This
standalone path does not modify the Atlas browser or its measurement importer.

## Deliverables and acceptance

- [X] Named-material listing/resolution and hash-checked prepared-asset registry.
- [X] Reviewed Hostun source acquisition/preparation with versioned cache identity.
- [ ] Source/scale recipes for additional named materials.
- [X] Case-file runner, common CLI overrides and deterministic dry-run resolution.
- [X] One output directory with resolved case, manifest, history, checkpoint and exports.
- [X] Typed synthetic virtual-test projection, preparation audit and raw-observation provenance.
- [X] Actual Atlas converter/browser recorded-request validation with explicit synthetic metadata.
- [X] Standalone native replay/comparison command with explicit synthetic JSON and
  optional SVG/PNG overlays, compiled-catalogue identity and ordered observations.
- [ ] Dedicated Atlas browser synthetic-evidence overlay import and UI labeling;
  standalone comparisons are available, while browser overlays remain unsupported.
- [X] End-to-end synthetic-fixture execution and failed-preparation tests on Metal.
- [ ] Qualified named Hostun preparation/loading study; the two-step named-material
  smoke run checks execution only. Contact and material qualification remain open.
