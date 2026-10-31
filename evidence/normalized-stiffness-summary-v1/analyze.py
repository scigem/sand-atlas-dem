"""Retrospective stiffness, overlap and timestep analysis under unchanged gates."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sand_atlas_dem.util import read_json,sha256,write_json

out=Path(__file__).resolve().parent
base=out.parent
short=base/'normalized-stiffness-contacts-v2'
long=base/'normalized-stiffness-contacts-long-v1'
paths=[short/'report.json',long/'report.json',short/'source_snapshot/studies/normalized-stiffness-contacts-v2.json',long/'source_snapshot/studies/normalized-stiffness-contacts-long-v1.json',Path(__file__)]
paths.extend(sorted(short.glob('*-k*-dt*.json')))
paths.extend(sorted(long.glob('*-k*-dt*.json')))
hashes={str(p):sha256(p) for p in paths}
write_json(out/'prospective-freeze.json',dict(files=hashes,scope='Retrospective analysis; no new gates'))
a,b=read_json(short/'report.json'),read_json(long/'report.json')
policy=read_json(paths[2]);long_policy=read_json(paths[3])
assert a['sources_unchanged'] and b['sources_unchanged'] and a['all_cases_recorded'] and b['all_cases_recorded']
for folder in [short,long]:
 f=read_json(folder/'prospective-freeze.json')['files']
 assert all(sha256(folder/'source_snapshot'/p)==v for p,v in f.items())
rows=[]
for c in a['cases']:
 s=c['static']
 rows.append(dict(fixture=c['fixture'],kappa=c['kappa'],stiffness_N_m3=s['normal_stiffness_N_m3'],overlap_fraction=s['reference_overlap_fraction'],production_overlap_comparison_passed=s['production_overlap_comparison_passed'],quadrature_passed=s['coarse_quadrature_error']<=policy['maximum_coarse_quadrature_error'] and s['reference_quadrature_change']<=policy['maximum_reference_quadrature_change'],gpu_static_passed=max(s['gpu_oracle_force_error'],s['gpu_oracle_torque_error'])<=policy['maximum_static_gpu_oracle_error'],local_normal_period_us=s['local_normal_period_s']*1e6,short_reference_supported=c['reference_supported_by_2ns'],short_largest_tested_accurate_ns=c['largest_tested_accurate_timestep_s']*1e9,short_upper_sweep_boundary_reached=c['largest_tested_accurate_timestep_s']==max(policy['timesteps_s'])))
long_rows=[]
for c in b['cases']:
 reference_file=long/f"{c['fixture']}-k{c['kappa']}-dt{long_policy['reference_timestep_s']:.0e}.json"
 r=read_json(reference_file)['rows']
 q0=np.array(r[0]['rotation']);maxangle=0.
 for observation in r:
  q=np.array(observation['rotation']);q/=np.linalg.norm(q,axis=1)[:,None]
  q_initial=q0/np.linalg.norm(q0,axis=1)[:,None]
  distance=np.minimum(np.linalg.norm(q-q_initial,axis=1),np.linalg.norm(q+q_initial,axis=1))
  maxangle=max(maxangle,float(np.max(4*np.arcsin(np.clip(distance/2,0,1)))))
 long_rows.append(dict(fixture=c['fixture'],kappa=c['kappa'],reference_supported=c['reference_supported_by_next_timestep'],largest_tested_accurate_ns=c['largest_tested_accurate_timestep_s']*1e9 if c['largest_tested_accurate_timestep_s'] else None,comparisons=c['comparisons'],observed_rotation_excursion_rad=maxangle,final_friction_loss_J=r[-1]['friction_loss_J'],maximum_observed_overlap=c['reference_maximum_observed_overlap']))
fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
for fixture in policy['fixtures']:
 values=[r for r in rows if r['fixture']==fixture]
 axes[0].loglog([r['kappa'] for r in values],[100*r['overlap_fraction'] for r in values],'.-',label=fixture)
 for kappa,style in [(1000,'-'),(30000,'--')]:
  c=next(c for c in long_rows if c['fixture']==fixture and c['kappa']==kappa)
  x=[v['timestep_s']*1e9 for v in c['comparisons'][1:] if 'maximum_errors' in v]
  y=[max(v['maximum_errors'][key]/limit for key,limit in policy['accuracy_limits'].items()) for v in c['comparisons'][1:] if 'maximum_errors' in v]
  axes[1].loglog(x,y,style,marker='.',label=f'{fixture}; κ={kappa}')
axes[0].axhline(1,color='black',ls=':',label='1% production comparison')
axes[0].set(xlabel='κ = traction stiffness × diameter / reference pressure',ylabel='Maximum static penetration / diameter (%)',title='Same declared normal force scale')
axes[1].axhline(1,color='black',ls=':',label='Frozen accuracy threshold')
axes[1].set(xlabel='Timestep (ns)',ylabel='Largest normalized error / its tolerance',title='65.536 µs Metal trajectories vs 8 ns')
for ax in axes:
 ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Synthetic grain-pair stiffness sweep; no assembly or material admission',fontsize=11)
fig.tight_layout();fig.savefig(out/'stiffness-and-timestep.png',dpi=160)
assert all(sha256(p)==v for p,v in hashes.items())
report=dict(identity=out.name,inputs_unchanged=True,static_cases=rows,long_cases=long_rows,short_trajectories=len(a['cases'])*len(policy['timesteps_s']),long_trajectories=len(b['cases'])*len(long_policy['timesteps_s']),scientific_admission=False)
write_json(out/'report.json',report)
print(json.dumps(dict(static_cases=rows,long_cases=[{k:v for k,v in c.items() if k!='comparisons'} for c in long_rows]),indent=2))
