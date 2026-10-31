"""Retrospective read-only trajectory diagnostic, with no new acceptance threshold."""

import json
from pathlib import Path

import h5py
import numpy as np

from sand_atlas_dem.util import read_json, sha256, write_json

output = Path(__file__).resolve().parent
study = output.parent / 'hostun-balanced424-preparation-v3/study'
chunk = study / 'run/chunk-0000'
checkpoint = chunk / 'checkpoint.h5'
manifest = read_json(chunk / 'manifest.json')
inputs = {str(p): sha256(p) for p in (checkpoint, chunk / 'history.json', chunk / 'manifest.json', Path(__file__))}
with h5py.File(checkpoint) as f:
    spec = json.loads(f.attrs['experiment'])
    initial_cell = np.array(json.loads(f.attrs['initial_cell']))
    accepted_stage = int(f.attrs['stage_index'])
    contact_phase = bool(f.attrs['contact_phase'])
    checkpoint_step = int(f.attrs['step'])
rows = read_json(chunk / 'history.json')
with h5py.File(spec['shape_library']) as f:
    volumes = [f[k].attrs['volume'] for k in sorted(f, key=int)]
# The frozen balanced constructor has exactly eight copies of each source grain.
if len(volumes) != 53 or spec['particles'] != 424 or spec['shape_sampling'] != 'balanced':
    raise ValueError('Unexpected physical inventory')
solid_volume = sum(volumes) * 8 * spec['scale']**3
steps = np.array([r['step'] for r in rows])
times = steps * spec['timestep']
cell = np.array([r['cell'] for r in rows])
rate = -min(spec['stages'][0]['gain'], spec['stages'][0]['max_rate'])
expected_cell = initial_cell * np.exp(rate * times[:, None])
actual_phi = solid_volume / np.prod(cell, axis=1)
actual_void = np.array([r['void_ratio'] for r in rows])
if any(sha256(path) != digest for path, digest in inputs.items()):
    raise ValueError('Completed checkpoint evidence changed during observation')
write_json(output / 'report.json', dict(
    identity=output.name, retrospective_diagnostic=True, acceptance_policy_added=False,
    preparation_accepted=False, scientific_admission=False, source_unchanged=True,
    input_sha256=inputs, step=checkpoint_step, samples=len(rows),
    ordered_complete_diagnostic_steps=bool(np.array_equal(steps,np.arange(0,250001,125))),
    physical_time_s=float(times[-1]), contact_phase=contact_phase, accepted_stage=accepted_stage,
    maximum_active_grain_pairs=max(r['active_grain_pairs'] for r in rows),
    maximum_pressure_Pa=max(r['p'] for r in rows),
    maximum_kinetic_energy_J=max(r['kinetic_energy'] for r in rows),
    maximum_spring_proxy_energy_J=max(r['elastic_energy_proxy'] for r in rows),
    maximum_relative_cell_difference_from_prescribed_exp=float(np.max(abs(cell/expected_cell-1))),
    maximum_absolute_void_ratio_difference_from_rigid_volume=float(np.max(abs(actual_void-(1/actual_phi-1)))),
    initial_cell_m=initial_cell.tolist(), final_cell_m=cell[-1].tolist(),
    initial_nominal_solid_fraction=float(actual_phi[0]), final_nominal_solid_fraction=float(actual_phi[-1]),
    estimated_device_working_bytes=manifest['estimated_memory_bytes'],
    cumulative_advance_wall_seconds=sum(r.get('wall_seconds',0) for r in rows),
    note='Observation of the first immutable completed chunk only. No thresholds are fitted or gates closed. Zero force/energy before contacts is not equilibrium acceptance. Timing is a dilute actual-workload observation, not dense or sustained benchmark throughput.'))
