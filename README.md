# Sand Atlas DEM

A standalone research implementation of mesh-derived, surface-based DEM
in **Slang**, executed through explicit **SlangPy/Metal** or a native **CPU** backend.
There is no automatic backend fallback. Python prepares geometry, schedules tests
and writes evidence. The selected backend executes neighbour search, contacts,
friction history and rigid-body updates.

**Status:** working GPU and CPU solvers and laboratory driver; scientific pilot acceptance
is still open. See [STATUS.md](docs/STATUS.md), the [implementation board](IMPLEMENTATION_PLAN.md)
and the [contact-method contract](docs/CONTACT_METHOD.md). A fast benchmark is not
experimental validation or a verified soil specimen.

Research prerelease: **v0.1.0a1**. See [release notes](RELEASE_NOTES.md).

## Setup

Requires Apple Silicon, macOS 26+, Python 3.12 and Xcode's Metal compiler toolchain.
The current SlangPy dependency is pinned to 0.43.1. The Metal backend is experimental.

The CPU backend uses the bundled Slang compiler and a C++17 compiler (`xcrun
clang++` on macOS, `CXX` or `c++` on Linux). CPU execution does not create a Metal
device. Apple Silicon is verified locally; Linux portability is not yet tested.
See [CPU execution and parallel jobs](docs/CPU_SOLVER.md).

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
An additional [wall example](docs/WALL_CONTACT.md) runs prescribed translating
planes with nonperiodic neighbours, wall reactions/work and restartable state:
`sand-dem run examples/wall-smoke.json --output runs/wall-smoke`.
`examples/pilot-template.json` specifies a 10,000-grain preparation/triaxial study,
**with provisional controller settings**, not a validated ready-to-publish test.
Preparation that exhausts its budget without convergence is a failed run.

## Live 3D view and videos

Install the optional renderer once:

```sh
.venv/bin/python -m pip install -e '.[view]'
.venv/bin/sand-dem run examples/smoke.json --output runs/view-smoke \
  --live --record-video --color-by force --vectors velocity
# The same flags work on resume and case run:
.venv/bin/sand-dem case run hostun --case cases/drained-triaxial.json \
  --output runs/hostun-video --record-video --view-every 10 \
  --color-by coordination --view-geometry sphere
```

`--live` opens an interactive 3D window (drag to rotate, wheel to zoom).
`--record-video` writes four 960×720 MP4s: `x.mp4`, `y.mp4`, `z.mp4` and
`iso.mp4` in `<run>/visualization/` (`<case-output>/run/visualization/` for cases).
Both flags can be combined. The three axis views are orthographic, looking along
the named axis toward the cell centre; videos retain a fixed initial-cell framing.
Closing the live window lets the simulation and any recording continue.

`--color-by` accepts `speed`, `force`, `torque`, `momentum`, `coordination` and
`shape`. `--vectors` accepts `velocity`, `force`, `torque` and `momentum`.
Labels use SI units; speed and velocity are peculiar motion, excluding affine
cell streaming. Force and torque are net particle contact resultants, so balanced
contact forces can cancel; this viewer does not show individual nodal force chains.
Coordination counts distinct active grain neighbours across both directed passes.
The white box is the periodic cell, with no ghost copies or clipping of crossing grains.

Rendering samples committed states at diagnostic boundaries. `--view-every N`
selects every Nth chunk; initial and final committed states are included.
`--video-fps 24` sets playback speed, independently of physical simulation time;
each frame displays its step and physical time. Default colours rescale each frame;
use `--color-max VALUE` to fix the zero-to-maximum range for comparisons.
Arrows show direction and relative magnitude with a per-frame display scale;
`--max-vectors 1000` limits arrows by deterministic striding.

Original mesh geometry is the default. `--view-geometry sphere` uses bounding
spheres as an explicit display proxy for large assemblies. Rendering and GPU
readback add wall time, especially for detailed meshes and four videos; they do
not change the solver timestep, diagnostic cadence or integration.
`render.json` records options, frame count and rendering failures. A display/encoder
error stops rendering with a warning while the simulation continues; check rendering
status separately from the run's scientific and execution status. Existing snapshots
and checkpoints retain their formats. PyVista/VTK renders the scene; ImageIO's
bundled FFmpeg encodes H.264. macOS rendering requires access to a graphical session.

## Named-material cases

```sh
# Optional explicit first-use preparation (also automatic on case run):
.venv/bin/sand-dem material prepare hostun --workers 4
.venv/bin/sand-dem case run hostun --case cases/drained-triaxial.json --dry-run
.venv/bin/sand-dem case run hostun --case cases/drained-triaxial.json \
  --pressure 20000 --rate 0.5 --output runs/hostun-tc
```

Hostun acquisition uses a reviewed source/scale recipe and private prepared cache.
Other materials currently require manual preparation and registration.
Material names resolve hash-pinned geometry; contact properties remain explicit in
reusable case files. The triaxial template has provisional preparation settings.
The runner automatically writes resolved inputs, a run/checkpoints and synthetic
CSV/JSON exports, preserving failure status. An optional [Atlas bridge](docs/ATLAS_BRIDGE.md) also produces native recorded-path
requests and explicitly synthetic comparison data. The measurement overlay importer
is not used. See the
[case workflow](docs/CASE_WORKFLOW.md) for registration, overrides and open gates.

## Measured grain geometry

```sh
.venv/bin/sand-dem geometry prepare grain.stl grain.h5 --unit mm --resolution 64 --samples 2048
.venv/bin/sand-dem geometry inspect grain.h5 --output runs/grain-inspection
.venv/bin/sand-dem geometry face-quadrature grain.h5 grain-gauss3-l2.h5 --rule gauss3 --subdivisions 2
```

Fixed [per-grain quadrature selection](docs/ADAPTIVE_QUADRATURE.md) is available
through `geometry adaptive-quadrature` once a complete reference study passes.
It preserves physical geometry and keeps GPU/material qualification separate.

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
# Include the actual Atlas converter and native comparison checks:
SAND_DEM_METAL=1 SAND_DEM_ATLAS_ROOT=../constitutive-atlas \
  SAND_DEM_ATLAS_SOLVER=../constitutive-atlas/build/atlas_solver \
  SAND_DEM_ATLAS_CATALOGUE=../constitutive-atlas/build/generated/models.json \
  .venv/bin/python -m pytest -q
.venv/bin/python examples/contact_refinement.py
.venv/bin/sand-dem qualify studies/mechanics-v2.json --output runs/mechanics
.venv/bin/sand-dem benchmark --particles 10000 --resolution 64 --samples 2048 --steps 20 --repeats 3 --output runs/benchmark-10000
```

Current checkpoints retain compensated FP32 state: /3 for undamped periodic or
bounded domains, /4 for undamped planes, and /5 when normal traction viscosity is
enabled. See [PRECISION.md](docs/PRECISION.md) and
[contact damping](docs/CONTACT_DAMPING_DESIGN.md). `normal_viscosity` is explicit
in N·s/m³ for grain contacts and independently for each plane; its default is zero.
`examples/wall-viscous-smoke-v2.json` is a run/resume/export example with positive
viscous wall loss. It is a short software check, not a calibrated sand specimen.

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
and VTK geometry for offline inspection. The optional recorded-path adapter uses
Atlas’s actual converter with an explicit projection policy and model configuration.
The separate `sand-dem atlas compare` command executes a prepared handoff through
an existing matching Atlas native solver/catalogue and produces an ordered synthetic
comparison with optional SVG/PNG overlays; see [the bridge workflow](docs/ATLAS_BRIDGE.md).
Constitutive Atlas is not modified and its measured-data importer is not reused. Stress is compression-positive; engineering,
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

An opt-in [active-contact timestep policy](docs/CONTACT_TIMESTEP_POLICY.md) uses
current contact areas and rotational mobility instead of the whole surface. Set
`"timestep_policy": "active-contact"` in a complete experiment/case. Device checks
reject unsafe chunks with rollback; this does not automatically choose a timestep
or qualify a material. The default remains `"conservative"`.

Packed mesh geometry now has a verified content-addressed disk cache for faster
repeated case startup. See [cache controls and performance studies](docs/PERFORMANCE_FOLLOWUP.md).
Set `SAND_DEM_MESH_CACHE=off` to disable it, or use an explicit cache directory.
