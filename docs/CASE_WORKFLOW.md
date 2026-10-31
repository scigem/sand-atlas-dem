# Named-material case workflow

The named-material runner is implemented. Automatic source acquisition, versioned
numerical profiles and direct Atlas synthetic-test import remain open. The existing
`run`, `resume`, geometry and export commands remain available as lower-level tools.

## User interface

After registering a prepared material once:

```sh
sand-dem material register cases/hostun-material.json
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
The included Hostun receipt refers to the separately prepared private asset; a fresh
checkout must acquire/prepare that asset before registering it. Registration is not
an automatic downloader or a scientific qualification.

The case declares particle count, seed, density, contact stiffness and friction,
preparation pressure/history, loading rate and stopping condition. Numerical
settings may come from a named, versioned profile, with explicit overrides. Every
resolved value goes into the output manifest. A material name supplies grain
geometry and source identity; it does not supply calibrated mechanical properties.

`sand-dem-case/1` case files contain an `experiment` object without a geometry path.
Density, both contact stiffnesses, friction, timestep and stages must be explicit.
`cases/drained-triaxial.json` is a provisional research template, not a qualified
Hostun preparation recipe; `cases/mesh-hold-smoke.json` is a two-step execution check.

Reusable case files are the primary reproducible interface. Equivalent command
line overrides should cover common pressure, rate, strain, particle-count and seed
sweeps. A dry-run command resolves all settings and reports the step/physical-time budgets,
asset size and configured memory limit without starting Metal. Case-specific wall
time is explicitly unknown; the memory limit is not an allocation estimate. Low-level geometry and solver settings remain available.

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
measurement importer. A reviewed synthetic-test import contract is required;
do not disguise DEM output as measured evidence to satisfy that importer.

The handoff must retain material/geometry identity, parameters, preparation,
case and solver hashes, run status and qualification status. It needs explicit SI
units, compression-positive signs, tensor frame and strain measures. For a drained
triaxial projection, map DEM axial z to Atlas axial x and compute signed axial-minus-
mean-radial stress from the stress tensor; DEM's nonnegative J2 q is a different
quantity. Preserve the full tensor and radial asymmetry alongside any projected
curve. Finite-volume strain and engineering/logarithmic strain remain distinguished.

The initial supported output is a dry drained-triaxial virtual experiment with
stress, axial/radial/volume strain, void ratio and physical time. Isotropic and
general strain paths need their own admitted protocol mappings. Effective stress
interpretation, axisymmetric projection and continuum kinematic compatibility
must be declared. Successful import does not establish material calibration or
experimental validation.

## Deliverables and acceptance

- [X] Named-material listing/resolution and hash-checked prepared-asset registry.
- [ ] Automatic source acquisition/preparation with complete cache identity.
- [X] Case-file runner, common CLI overrides and deterministic dry-run resolution.
- [X] One output directory with resolved case, manifest, history, checkpoint and exports.
- [ ] Synthetic Atlas dataset plus metadata for preparation and full DEM observations.
- [ ] Validation against the actual Atlas consumer, including synthetic labels.
- [X] End-to-end synthetic-fixture execution and failed-preparation tests on Metal.
- [ ] Qualified named Hostun preparation/loading study; the two-step named-material
  smoke run checks execution only. Contact and material qualification remain open.
