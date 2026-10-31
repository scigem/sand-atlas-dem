"""Conditional remaining-cost diagnostic, with no acceptance gate or ETA promise."""
import json
import math
from pathlib import Path

import h5py
import numpy as np

from sand_atlas_dem.util import read_json, sha256, write_json

out=Path(__file__).resolve().parent
root=out.parents[1]
chunk=root/'evidence/hostun-balanced424-preparation-v3/study/run/chunk-0001'
policy_path=root/'studies/hostun-balanced424-preparation-policy-v3.json'
paths=[chunk/'checkpoint.h5',chunk/'history.json',policy_path,Path(__file__)]
hashes={str(p):sha256(p) for p in paths}
write_json(out/'prospective-freeze.json',dict(files=hashes,scope=__doc__,method='For rigid total volume and a latched per-axis compression-rate cap c, nominal phi can grow no faster than exp(3*c*t) in the continuous model. Compute log(phi_min/phi_now)/(3*c), then divide by dt. Ignore arithmetic roundoff; do not infer packing feasibility, pressure convergence or future step cost. Runtime scenarios apply observed phase step costs unchanged, not as an upper or lower wall-time bound.'))
policy=read_json(policy_path)
with h5py.File(chunk/'checkpoint.h5') as f:
    spec=json.loads(f.attrs['experiment'])
    assert bool(f.attrs['contact_phase']) and int(f.attrs['stage_index'])==0
    step=int(f.attrs['step'])
rows=read_json(chunk/'history.json')['rows']
last=rows[-1]
assert last['step']==step==500000
stage=spec['stages'][0]
assert stage['kind']=='isotropic' and spec['periodic'] and not spec['planes']
cap=min(stage['max_rate'],stage['contact_max_rate'])
phi=1/(1+last['void_ratio'])
physical_minimum=max(0.,math.log(policy['minimum_final_solid_fraction']/phi))/(3*cap)
steps_minimum=physical_minimum/spec['timestep']
first=next(i for i,r in enumerate(rows) if r['active_surface_contacts']>0)
# Each row's wall time belongs to the interval ending at that row. The slower
# latched cap first governs the interval after the first contact observation.
phase=rows[first+1:]
assert phase and all(np.all(abs(np.array(r['cell_log_rate']))<=cap) for r in phase)
steps=np.array([r['step']-rows[first+i]['step'] for i,r in enumerate(phase)],dtype=float)
seconds=np.array([r['wall_seconds'] for r in phase])
costs=seconds/steps
mean=float(seconds.sum()/steps.sum())
scenarios=[]
for name,value in [('observed-post-latch-mean',mean),('observed-post-latch-median',float(np.median(costs))),('observed-post-latch-p95',float(np.quantile(costs,.95)))]:
    scenarios.append(dict(cost_basis=name,seconds_per_step=value,wall_hours_for_minimum_density_steps=value*steps_minimum/3600))
assert all(sha256(p)==v for p,v in hashes.items())
write_json(out/'report.json',dict(identity=out.name,input_sha256=hashes,inputs_unchanged=True,retrospective_diagnostic=True,acceptance_policy_added=False,gpu_executed=False,preparation_accepted=False,scientific_admission=False,checkpoint_step=step,checkpoint_physical_time_s=last['elapsed_time'],nominal_solid_fraction=phi,minimum_final_solid_fraction=policy['minimum_final_solid_fraction'],latched_per_axis_compression_cap_s_inv=cap,maximum_continuous_log_solid_fraction_growth_s_inv=3*cap,continuous_model_minimum_additional_physical_time_s=physical_minimum,approximate_minimum_additional_steps_ignoring_roundoff=math.ceil(steps_minimum),minimum_density_first_diagnostic_step_ignoring_roundoff=step+math.ceil(steps_minimum/spec['diagnostic_interval'])*spec['diagnostic_interval'],declared_remaining_step_budget=stage['steps']-step,declared_remaining_physical_time_budget_s=(stage['steps']-step)*spec['timestep'],post_latch_observed_steps=int(steps.sum()),post_latch_observed_wall_seconds=float(seconds.sum()),runtime_scenarios=scenarios,limitations=['The physical bound concerns only the nominal 0.45 density floor, not stable packing, 10kPa pressure, contact inventory or a 100-sample equilibrium window.','Continuous-model arithmetic ignores FP32/log-state roundoff, so the integer diagnostic step is approximate.','Scenarios hold observed dilute first-contact step cost fixed. Future dense contacts, allocation growth, compilation, thermals and interruptions may change it. These are not measured dense benchmark results or an ETA.','No automatic controller rate increase, step-budget extension, gate revision or restart is authorized by this diagnostic.']))
print(json.dumps(dict(minimum_additional_ms=physical_minimum*1000,minimum_additional_steps=steps_minimum,scenarios=scenarios),indent=2))
