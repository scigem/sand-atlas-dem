"""Read-only completed-interval diagnostics; no new acceptance gate or GPU work."""
import json
from pathlib import Path

import h5py
import numpy as np

from sand_atlas_dem.util import read_json, sha256, write_json

output = Path(__file__).resolve().parent
study = output.parent / 'hostun-balanced424-preparation-v3/study'
chunk = study / 'run/chunk-0003'
checkpoint = chunk / 'checkpoint.h5'
paths = [checkpoint, chunk/'history.json',chunk/'manifest.json',study/'frozen-manifest.json',Path(__file__)]
# The frozen manifest's running status can change at completion; snapshot its bytes
# for this observation rather than require the live parent status to remain fixed.
manifest_snapshot = output/'parent-frozen-manifest-observed.json'
manifest_snapshot.write_bytes((study/'frozen-manifest.json').read_bytes())
paths[-2] = manifest_snapshot
inputs = {str(p):sha256(p) for p in paths}
frozen = read_json(manifest_snapshot)
archive_verified = all(sha256(study/'source_snapshot'/name)==digest for name,digest in frozen['sources'].items())
if not archive_verified:
    raise ValueError('Accepted numerical source archive changed')
with h5py.File(checkpoint) as f:
    spec = json.loads(f.attrs['experiment'])
    initial_cell = np.array(json.loads(f.attrs['initial_cell']))
    checkpoint_meta = {name:int(f.attrs[name]) for name in ('step','stage_index','stage_steps','converged','contacts')}
    checkpoint_contact_phase=bool(f.attrs['contact_phase'])
    checkpoint_rows=json.loads(f.attrs['rows'])
    meta=f['meta'][:]
    normals=f['normalArea'][:]
    slip=f['slip'][:]
rows=read_json(chunk/'history.json')['rows']
assert rows==checkpoint_rows
assert checkpoint_meta['step']==1000000 and len(rows)==8001
assert np.array_equal([r['step'] for r in rows],np.arange(0,1000001,125))
assert sha256(Path(spec['shape_library'])) == frozen['shape_sha256']
with h5py.File(spec['shape_library']) as f:
    groups=[f[k] for k in sorted(f,key=int)]
    assert len(groups)==53 and spec['particles']==424 and spec['shape_sampling']=='balanced'
    volume=sum(g.attrs['volume'] for g in groups)*8*spec['scale']**3
    sample_counts=np.array([len(g['samples']) for g in groups])
ids=np.random.default_rng(spec['seed']).permutation(np.arange(424,dtype=np.uint32)%53)
valid_body_keys=bool(np.all(meta[:,:2]<424) and np.all(meta[:,0]!=meta[:,1]) and np.all(meta[:,3]==0))
assert valid_body_keys
valid_sample_keys=bool(np.all(meta[:,2]<sample_counts[ids[meta[:,0]]]))
unique_keys=len(np.unique(meta,axis=0))==len(meta)
first=next(i for i,r in enumerate(rows) if r['active_surface_contacts']>0)
filter_stress=np.zeros(3)
latched=False
stage=spec['stages'][0]
dt=spec['diagnostic_interval']*spec['timestep']
alpha=-np.expm1(-dt/stage['filter_time'])
expected_rates=[]
for previous in rows[:-1]:
    filter_stress+=alpha*(np.diag(previous['stress'])-filter_stress)
    latched=latched or previous['active_surface_contacts']>0
    cap=min(stage['max_rate'],stage['contact_max_rate']) if latched else stage['max_rate']
    expected_rates.append(np.clip(stage['gain']*(filter_stress-stage['pressure'])/stage['pressure'],-cap,cap))
rates=np.array([r['cell_log_rate'] for r in rows[1:]])
expected_log=np.cumsum(-rates*dt,axis=0)
cells=np.array([r['cell'] for r in rows])
phi=volume/np.prod(cells,axis=1)
void=np.array([r['void_ratio'] for r in rows])
last=rows[-1]
assert all(sha256(p)==digest for p,digest in inputs.items())
report=dict(identity=output.name,retrospective_diagnostic=True,acceptance_policy_added=False,preparation_accepted=False,scientific_admission=False,gpu_executed=False,reanalyzed_existing_gpu_evidence=True,inputs_unchanged=True,archived_solver_verified=archive_verified,input_sha256=inputs,checkpoint=checkpoint_meta,checkpoint_contact_phase=checkpoint_contact_phase,history_samples=len(rows),ordered_complete_diagnostics=True,first_contact_step=rows[first]['step'],first_contact_time_s=rows[first]['elapsed_time'],first_contact_surface_nodes=rows[first]['active_surface_contacts'],slower_rate_first_applied_at_step=rows[first+1]['step'],maximum_absolute_controller_rate_replay_difference_s_inv=float(np.max(abs(rates-np.array(expected_rates)))),maximum_relative_cell_difference_from_integrated_logged_rates=float(np.max(abs(cells[1:]/(initial_cell*np.exp(-expected_log))-1))),maximum_absolute_void_difference_from_rigid_volume=float(np.max(abs(void-(1/phi-1)))),maximum_observed_overlap_fraction=max(r['max_overlap_fraction'] for r in rows),declared_overlap_guard_fraction=spec['max_overlap_fraction'],active_checkpoint_keys_valid=valid_body_keys and valid_sample_keys and unique_keys,active_checkpoint_array_lengths_match=bool(len(meta)==len(normals)==len(slip)==checkpoint_meta['contacts']==last['active_surface_contacts']),maximum_normal_unit_length_difference=float(np.max(abs(np.linalg.norm(normals[:,:3],axis=1)-1))),all_checkpoint_history_values_finite=bool(np.isfinite(normals).all() and np.isfinite(slip).all()),final_nominal_solid_fraction=float(phi[-1]),final_pressure_Pa=last['p'],target_pressure_Pa=stage['pressure'],final_active_grain_pairs=last['active_grain_pairs'],required_final_active_grain_pairs=424,final_active_surface_contacts=last['active_surface_contacts'],final_kinetic_to_proxy_ratio=last['kinetic_energy']/last['elastic_energy_proxy'],final_unbalanced_force=last['unbalanced_force'],cumulative_observed_advance_seconds=sum(r.get('wall_seconds',0) for r in rows),observation_note='First dilute contacts, not a dense equilibrated specimen. Controller replay and geometric identities are diagnostics, with no new fitted thresholds. This does not replace terminal independent preparation acceptance or mesh work/timestep studies. Timings include varying dilution/contact phases and are not a dense or sustained benchmark.')
previous_chunk = study/'run/chunk-0002'
previous_rows = read_json(previous_chunk/'history.json')['rows']
assert rows[:len(previous_rows)] == previous_rows
interval = rows[len(previous_rows):]
costs = np.array([r['wall_seconds']/125 for r in interval])
report.update(previous_checkpoint_prefix_preserved=True,
              incremental_steps=250000,
              incremental_advance_seconds=float(sum(r['wall_seconds'] for r in interval)),
              incremental_mean_seconds_per_step=float(np.mean(costs)),
              incremental_median_seconds_per_step=float(np.median(costs)),
              incremental_p95_seconds_per_step=float(np.percentile(costs,95)),
              checkpoint_sha256=sha256(checkpoint), shape_sha256=frozen['shape_sha256'])
write_json(output/'report.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','observation_note')},indent=2))
