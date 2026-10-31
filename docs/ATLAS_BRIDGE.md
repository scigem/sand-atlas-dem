# Synthetic DEM → Constitutive Atlas recorded paths

The bridge produces a request accepted by the existing Atlas Python and browser
**recorded-path** interfaces. It also exports a distinctly typed synthetic virtual
experiment for comparison. Neither adjacent repository is modified. The current
measurement importer is unsuitable for DEM overlays and is not used.

## Commands

For an existing completed preparation/loading run:

```sh
sand-dem atlas prepare runs/hostun-tc/run \
  --atlas-root ../constitutive-atlas \
  --policy cases/atlas-projection.json \
  --configuration cases/atlas-elastic-replay.json \
  --output runs/hostun-tc/atlas
```

Or attach the handoff to the case run:

```sh
sand-dem case run hostun --case cases/drained-triaxial.json \
  --stop-strain 0.005 --output runs/hostun-small-tc \
  --atlas-root ../constitutive-atlas \
  --atlas-policy cases/atlas-projection.json \
  --atlas-config cases/atlas-elastic-replay.json
```

The policy and continuum-model configuration are frozen in the resolved case before
GPU initialization. Dry-run displays them without running the converter. The example
linear-elastic coefficients and initial `e=0.7` are illustrative, not a Hostun
calibration. Supply all actual continuum parameters and initial history explicitly.
Atlas rejects a contradictory initial void ratio unless its configuration contains
an explicit `initial_void_ratio_override_note`. A failed preparation is rejected;
a completed DEM run can separately fail handoff, and both statuses are retained.
The Hostun loading/preparation template remains scientifically unqualified.

Import `atlas/replay-request.json` in the matching Atlas build's **Recorded Paths**
page, or use its existing `atlas.cli replay` command with an appropriate solver.
The request prescribes the projected DEM axial-strain and mean radial-stress history
for the chosen continuum model. Creating a request does not execute that model.
The current page plots the model response; it does not overlay the DEM curve or
read the comparison sidecar automatically. Dedicated synthetic-evidence overlays
remain separate integration work. Use a matching local Atlas version/model catalogue;
this work does not verify a deployed website or every continuum model.

## Preconditions and rejection

- Completed `sand-dem-run/1`, exactly isotropic preparation then triaxial loading.
- Verified history, prepared checkpoint, shape-library identity and run artifacts.
- Prepared checkpoint identifies the actual final preparation cell and step.
- Independent audit of the final 100 ordered preparation diagnostics: each normal
  stress within 1% of the declared pressure, kinetic/elastic proxy ratio ≤1e-5,
  and unbalanced force ≤1e-3. Policy files cannot relax these gates.
- One through 5,000 loading observations, plus the preparation origin. No sorting,
  trimming, interpolation or automatic downsampling. Time and step remain ordered;
  duplicate/reversing strain observations are retained.
- Explicit dry/effective-stress interpretation, lossy axisymmetric projection,
  engineering-strain measure, caller's continuum approximation and strain bound.

The supplied policy limits each engineering-strain component to 1% relative to the
prepared state. This is a caller declaration, not proof of infinitesimal validity.
Cases outside their declared bound fail; a revised study needs another policy/output
identity and retains the earlier failure. The general 10% triaxial template therefore
cannot be handed off using the example 1% policy without a smaller loading target.

## Mapping and retained information

DEM's axial z becomes Atlas x using the right-handed order `[z,x,y]`. For each row:

- `sigma_a = sigma_zz`, `sigma_r = (sigma_xx + sigma_yy)/2`;
  `p = trace(sigma)/3`, **signed** `q = sigma_a - sigma_r`.
- Compression-positive engineering strains are `1 - cell_i/prepared_cell_i`.
  Radial strain is their x/y mean; volumetric strain is their sum.
- Physical time starts at the actual prepared observation. Atlas incremental replay
  instead advances by ordered observation index and does not use elapsed time.

`virtual-test.json` retains the full original loading observations, mapped stress
tensors, radial-stress difference, shear/antisymmetry diagnostics, logarithmic
strains and finite-volume strain `1 - V/V_prepared`. The last two volume measures
are distinct. DEM's nonnegative J2 q is retained separately and never substituted
for signed triaxial q. The original preparation history stays in the hash-linked
source run; retain the complete case directory for reproducible evidence.

The replay controls both radial stresses with `p-q/3` and fixes shear strain to zero.
This discards the original radial asymmetry and shear in the replay projection;
the raw data remain available. Because q contributes to the radial constraint,
comparing q afterward is not an independent input-free test. Nominal void ratio
ignores grain overlap; no fluid/pore-pressure model is inferred.

## Artifacts and verification scope

The output contains `virtual-test.json` (`kind: synthetic-dem`), native
`replay-request.json`, Atlas's own conversion provenance, frozen configuration/policy,
converter log and `handoff.json` with consumer source/model/schema hashes. It checks
that parsed inputs and consumer sources remain unchanged during conversion.
A private legacy-schema intermediate carries `provenance.evidence_kind: synthetic-dem`
and explicit warnings; it is never passed to the measurement importer.

`examples/atlas_bridge_study.py` freezes and checks an **analytic software fixture**
through the public CLI, actual Atlas Python converter and browser request validator.
It executes neither a DEM specimen nor a continuum solver. The algebra tests cover
signed q, full tensor permutation, prepared-state origins, strain measures, negative
cases and ordered reversals/duplicates. The prospective study is not evidence of
qualified material response or browser visual QA. Real qualified DEM loading,
continuum execution/comparison and physical experimental validation remain open.

## Native continuum replay verification

`atlas-native-replay-v2` runs the existing Atlas native solver on the five original
analytical bridge observations. The request, handoff artifacts, native binary,
consumer Python/schema/model bytes and study source are hash-frozen before
execution. Returned coordinates must exactly preserve every original knot,
including the reversal and duplicate. Nothing is sorted or interpolated.

An independent isotropic Hooke compliance calculation checks all six stress and
strain components at each observation; exponential void-ratio evolution is checked
separately. Maximum errors are 1.819e-12 Pa, 1.525e-20 strain and 2.221e-16 void
ratio, below the prospectively declared 0.002 Pa, 2e-11 and 2e-10 limits. V1 is
retained separately; v2 adds the explicit coordinate identity check with unchanged
numerical gates. No adjacent repository or binary was modified or rebuilt.

This proves actual native continuum execution for the analytical software fixture.
It does not turn that fixture into a GPU-produced specimen or establish agreement
for qualified Hostun loading. The native API returns model state and strain;
a generic comparison tool must resolve the model's explicit stress binding and
state layout, rather than assume the first six values are stress. This study
checks the pinned linear-elastic layout only. Dedicated Atlas browser overlays and
qualified material replay remain open; the standalone comparison command below is implemented.

## Execute and compare a prepared handoff

The standalone DEM CLI can now execute the handoff and write a dedicated synthetic
comparison without changing either atlas:

```sh
sand-dem atlas compare runs/hostun-tc/atlas \
  --atlas-root ../constitutive-atlas \
  --solver ../constitutive-atlas/build/atlas_solver \
  --catalogue ../constitutive-atlas/build/generated/models.json \
  --output runs/hostun-tc/comparison --plots
```

Supply an existing native solver and its generated catalogue. Nothing is built.
`--plots` requires the `inspect` extra and writes SVG and PNG overlays of signed q
and the sum of normal compression strains against axial compression strain.
The artifact distinguishes source engineering strain from continuum infinitesimal
strain and records the exact Mandel component order.
Without it, the complete ordered comparison is still written as JSON.

The command verifies handoff artifacts, the model definition hash from conversion,
the generated catalogue's canonical hash and parameter order, and the binary's
actual `--describe` catalogue. Atlas validates the model and executes its native
recorded replay. The response must identify the same request, binary catalogue,
parameters, initial state and every original source-row coordinate. Truncated and
failed responses cannot become completed comparisons; their native response/log
and a failed report are retained. Source, binary and input hashes are checked
again after execution. Reusing an output directory fails rather than overwriting.

Supported explicit v0.3 profiles must declare strain-path incremental/rate execution,
fixed spatial Mandel components, compression-positive signs, supported small-strain
kinematics and effective Cauchy stress bound to persistent state `sigma`.
Existing v0.1 records use Atlas's documented legacy incremental contract and retain
that identity in the report; no model is migrated. Physical-time observable stress,
total stress and absent/unsupported stress bindings reject. This matches the
current recorded-path consumer's capabilities. The model's actual state layout
and optional scalar void ratio are read from the matched compiled definition.
Void output requires exactly one persistent scalar whose explicit quantity is
`void_ratio`; absent or ambiguous semantics yield `null`. Legacy fields named `e`
remain in the raw native response rather than acquiring inferred semantics.

`comparison.json` contains every source observation, continuum stress/strain,
signed q and p, optional continuum void ratio, elapsed-time provenance and absolute
differences. Continuum strain changes are relative to the explicitly configured
initial strain. The raw handoff, full native response and compiled catalogue are
copied alongside the source freeze and `report.json`. No interpolation, sorting,
fitting or automatic scientific-admission decision occurs. Radial stress remains
a prescribed synthetic-history constraint; agreement is not independent validation.

`atlas-native-comparison-v3` passes this CLI on the five original analytical fixture
observations, including reversal and duplicate, and exports visually inspected
SVG/PNG plots. Worst q and p differences are 3.991e-9 Pa and 1.328e-9 Pa; engineering
volumetric-strain difference is 2.175e-16. V1's more crowded tick layout and v2's earlier metadata/labels are retained
under their own identities. These are software-fixture results, not GPU-generated sand
response. Qualified Hostun loading/comparison and Atlas browser overlay import/UI
remain open.

`atlas-v03-comparison-v1` separately exercises the explicit v0.3 `barodesy-sand`
strain-path-rate model through prepare and compare, with all five original ordered
analytical-source observations. Its maximum normalized control error is
3.1484e-11, below the frozen 1e-10 gate; the duplicate retains the exact state and
strain. The plots show the different nonlinear response instead of forcing a match
to the elastic source fixture. `cases/atlas-barodesy-sand-software-replay.json`
contains its complete, explicit example coefficients and initial history. These
are software inputs, not a calibration for Hostun or validation of this DEM method.
