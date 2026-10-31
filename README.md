# Sand Atlas DEM

A standalone research implementation of mesh-derived, surface-based DEM
in **Slang**, executed through **SlangPy/Metal** on Apple Silicon. It has no CPU
simulation fallback. Python prepares geometry, schedules tests and writes evidence;
neighbour search, contacts, friction history and rigid-body updates run on the GPU.

**Status:** working GPU solver and laboratory driver; scientific pilot acceptance
is still open. See [STATUS.md](docs/STATUS.md), the [implementation board](IMPLEMENTATION_PLAN.md)
and the [contact-method contract](docs/CONTACT_METHOD.md). A fast benchmark is not
experimental validation or a verified soil specimen.

## Setup

Requires Apple Silicon, macOS 26+, Python 3.12 and Xcode's Metal compiler toolchain.
The current SlangPy dependency is pinned to 0.43.1. The Metal backend is experimental.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,inspect]'
# If xcrun metal --version reports a missing component:
xcodebuild -downloadComponent MetalToolchain
.venv/bin/sand-dem doctor
```

`doctor` must identify a Metal adapter and pass an actual compiled compute test.
Shader warnings about unused atomic return values are currently harmless. SlangPy
reports no supported shader model on this device but successfully executes Metal;
the output retains this warning rather than suppressing it.

## First run

```sh
.venv/bin/python examples/create_fixtures.py
.venv/bin/sand-dem run examples/smoke.json --output runs/smoke
.venv/bin/sand-dem export runs/smoke --output runs/smoke-export
.venv/bin/sand-dem resume runs/smoke/checkpoint.h5 --output runs/resumed
# Original concave mesh fixture, using triangle queries:
.venv/bin/sand-dem run examples/mesh-smoke.json --output runs/mesh-smoke
```

The smoke case is deliberately dilute and short. It tests execution, checkpointing
and exports; it is not a preparation or triaxial acceptance study.
`examples/pilot-template.json` specifies a 10,000-grain preparation/triaxial study,
**with provisional controller settings**, not a validated ready-to-publish test.
Preparation that exhausts its budget without convergence is a failed run.

## Named-material cases

```sh
# Once the separately prepared Hostun asset is available:
.venv/bin/sand-dem material register cases/hostun-material.json
.venv/bin/sand-dem case run hostun --case cases/drained-triaxial.json --dry-run
.venv/bin/sand-dem case run hostun --case cases/drained-triaxial.json \
  --pressure 20000 --rate 0.5 --output runs/hostun-tc
```

Material names resolve hash-pinned geometry; contact properties remain explicit in
reusable case files. The triaxial template has provisional preparation settings.
The runner automatically writes resolved inputs, a run/checkpoints and synthetic
CSV/JSON exports, preserving failure status. Direct Constitutive Atlas synthetic
import is still required; current exports are comparison bundles. See the
[case workflow](docs/CASE_WORKFLOW.md) for registration, overrides and open gates.

## Measured grain geometry

```sh
.venv/bin/sand-dem geometry prepare grain.stl grain.h5 --unit mm --resolution 64 --samples 2048
.venv/bin/sand-dem geometry inspect grain.h5 --output runs/grain-inspection
```

Input coordinates need a source-supported unit. Exact coordinate welding decodes
STL triangle soup; there is no automatic hole filling, smoothing, component removal
or winding repair. Invalid solids are rejected. VDB density volumes are not accepted
as signed-distance fields. Use [Sand Atlas](https://sand-atlas.scigem.com/) for source
geometry and retain the source identity, rights and scale separately.

The first legacy reference mesh was rejected and remains excluded. The direct
archive subsequently yielded 53 valid solids; their physical scale is inferred from
a one-to-one match with the dimensioned source descriptor table. See the
[updated acquisition report](docs/HOSTUN_ACQUISITION_V2.md). Private assets retain
source hashes, units and the explicit inference. Higher-resolution and material
qualification remain separate from source acquisition.

## Verification and benchmarks

Run GPU compilation/tests **serially**:

```sh
.venv/bin/python -m pytest -q -m 'not metal'
SAND_DEM_METAL=1 .venv/bin/python -m pytest -q
.venv/bin/python examples/contact_refinement.py
.venv/bin/sand-dem qualify studies/mechanics-v2.json --output runs/mechanics
.venv/bin/sand-dem benchmark --particles 10000 --resolution 64 --samples 2048 --steps 20 --repeats 3 --output runs/benchmark-10000
```

Current checkpoints are /3 and include compensated FP32 accumulation state; older
checkpoints are deliberately rejected. See [PRECISION.md](docs/PRECISION.md).

The suite includes independent FP64 contact evaluation, analytical geometry,
periodic contact, friction history/restart, force and torque balance, failed-step
rollback, strict source hashes and an explicit failed preparation.
The independent sphere study compares against a continuum surface integral.
Seven frozen mechanics cases and a separate higher-spin timestep-refinement study
now pass; their scope and retained failures are in
[MECHANICS_QUALIFICATION.md](docs/MECHANICS_QUALIFICATION.md).

Short M4 Max measurements are in `evidence/`. The original scalar-field implementation
measured approximately 10 ms/step for 10,000 spheres at 64-cell/2,048-sample resolution.
These are short **wall-clock** measurements including encoding and synchronization.
Metal GPU timestamp retrieval returned `SLANG_E_NOT_AVAILABLE`; reports preserve this
failure and do not claim measured GPU-only stage timings.

## Output and atlas connection

Each run contains a manifest, synthetic history, checkpoints and scheduled snapshots.
Exports provide CSV, a distinctly typed synthetic comparison bundle, response plots
and VTK geometry for offline inspection. Constitutive Atlas is not modified and its
measured-data importer is not reused. Stress is compression-positive; engineering,
logarithmic and volume-ratio measures are explicitly distinguished.

An optional exact-feature triangle-BVH distance path passes bounded device checks;
see [MESH_QUERY.md](docs/MESH_QUERY.md). The existing SDF path failed a shallow
Hostun force-accuracy gate; see [HOSTUN_CONTACT_ACCURACY.md](docs/HOSTUN_CONTACT_ACCURACY.md).
Neither geometry throughput nor a surface-distance pass establishes force accuracy.

Set `"contact_geometry": "mesh"` in an experiment to use original-triangle queries.
The mesh implementation includes nearest-first traversal, conservative enclosure
culling and one-pass contact compaction. All 37 tests pass; ten fixed Hostun poses
match an independent triangle-force reference within 0.00965%. This comparison
holds the surface samples fixed; contact-quadrature convergence remains open.

The frozen enclosure/compaction M4 Max mesh workloads measure
**15.20 ms/step at 10k grains** and
**60.32 ms/step at 50k** (isolated contacting pairs, 53 reused shapes, 2,048 nodes
per shape). These imply about 4.2/16.8 hours per million steps before setup and
output. At the tested 2 ns timestep this advances 2 ms physically. Dense specimens
and refined contact quadrature require separate measurements.

No calibrated sand properties, measured-data validation, finite-strain continuum
compatibility, NVIDIA backend verification or scientific admission is claimed.
Code is MIT; third-party sources retain their own rights. No raw publication texts
or downloaded Sand Atlas geometry are bundled in version control.

The current collector also rejects conservative spatial groups of surface nodes,
retaining original sample keys. This extension passes the 46-test integrated device/reference suite
and all ten static Hostun force cases; its own sustained timings are pending.
