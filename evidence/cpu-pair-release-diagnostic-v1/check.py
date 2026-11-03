"""Classify the retained v5 benchmark topology failure; no altered pass gate."""
import hashlib
import json
from pathlib import Path

import numpy as np

from sand_atlas_dem.benchmark import controlled_pairs
from sand_atlas_dem.cpu import CPU
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment

root = Path(__file__).resolve().parents[1]
study = root / "cpu-solver-verification-v5"
output = Path(__file__).resolve().parent
library_path = study / "fixture.h5"
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
          [Path(__file__), library_path, study / "freeze.json"]}
(output / "freeze.json").write_text(json.dumps({
    "id": output.name, "inputs": hashes, "steps": 2048, "timestep_s": 1e-9,
    "purpose": "Diagnostic only: report free-release topology and separation; does not revise v5 failure",
    "gpu_dispatches": 0,
}, indent=2) + "\n")
library = ShapeLibrary.load(library_path)
p, cell, ids, rotations, _ = controlled_pairs(library, 32)
engine = Engine(library, Experiment(particles=32, contact_geometry="mesh", timestep=1e-9,
    pairs_per_particle=1, contacts_per_particle=4096, max_overlap_fraction=.01),
    p, cell, ids, np.ones(32), metal=CPU())
engine.buffers["rotation"].copy_from_numpy(rotations)
engine.initialize()
before = engine.diagnostics()
engine.advance(2048)
state = engine.state()
after = engine.diagnostics()
positions = (state["position"][:, :3].astype(float) + state["positionRemainder"][:, :3]) * engine.length_unit
initial_distance = np.linalg.norm(p[1::2] - p[::2], axis=1)
final_distance = np.linalg.norm(positions[1::2] - positions[::2], axis=1)
report = {
    "id": output.name, "backend": "cpu", "scientific_admission": False,
    "initial_active_grain_pairs": before["active_grain_pairs"],
    "final_active_grain_pairs": after["active_grain_pairs"],
    "initial_contacts": before["active_surface_contacts"],
    "final_contacts": after["active_surface_contacts"],
    "initial_pair_distances_m": initial_distance.tolist(),
    "final_pair_distances_m": final_distance.tolist(),
    "all_pairs_moved_apart": bool(np.all(final_distance > initial_distance)),
    "all_state_arrays_finite": all(np.isfinite(v).all() for v in state.values() if isinstance(v, np.ndarray)),
    "final_kinetic_energy_J": after["kinetic_energy"],
    "input_integrity": all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items()),
    "interpretation": "Freely moving compressed grains release; long continuous contact-count invariant is unsuitable. Original v5 remains failed.",
}
(output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
