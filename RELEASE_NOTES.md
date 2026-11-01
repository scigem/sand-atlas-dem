# v0.1.0a1 — research prerelease

First tagged snapshot of the Slang/Metal DEM laboratory for Apple Silicon.

- GPU neighbour search, mesh/SDF contact, friction history, rigid-body integration, periodic cells and translating plane boundaries.
- Compensated local coordinates, restart/checkpoint support, active-contact timestep guards, explicit pressure-normalized stiffness studies, CLI material/case workflows and offline Constitutive Atlas comparison.
- Production parallel mesh traversal and verified content-addressed mesh packing cache. Frozen Hostun measurements reduce packing from 97.24 s to 0.61 s on a cache hit; this is setup time, not simulation throughput.
- Retained deterministic gathering, immutable binding and spatial-index experiments, including unsuccessful equivalence and performance results. Experimental helpers remain opt-in examples; they do not change production dispatch or geometry defaults.

Requires Apple Silicon, macOS 26+, Python 3.12 and SlangPy 0.43.1. This is a research implementation: scientific material admission, full assembly convergence and reproduction of published benchmarks remain open. The Hostun preparation remains stopped. Downloaded geometry and large/private checkpoints are excluded.

The previous production regression passed 329 tests with eight skips. Post-push release qualification runs the full serial Metal/Atlas suite, independent sphere contact refinement and the unchanged mechanics-v2 protocol. Results will be attached to this release after execution; existing failed studies remain preserved.

Reproduce with `PYTHONPATH=src SAND_DEM_ATLAS_ROOT=../constitutive-atlas SAND_DEM_ATLAS_SOLVER=../constitutive-atlas/build/atlas_solver SAND_DEM_ATLAS_CATALOGUE=../constitutive-atlas/build/generated/models.json .venv/bin/python examples/qualify_release.py --output runs/release-v0.1.0a1`.
