"""Retrospective comparison; original trajectories and gates remain unchanged."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sand_atlas_dem.util import sha256,write_json

out=Path(__file__).resolve().parent
base=out.parent
names=['contact-timestep-limits-v1','contact-timestep-origin-diagnostic-v1','contact-local-frame-timesteps-v1','contact-local-frame-origin-v1']
paths=[base/n/'report.json' for n in names]
for n in names:
 paths.extend(sorted((base/n).glob('rebound-*.json')))
paths.extend([base/'contact-local-frame-v1/report.json',Path(__file__)])
hashes={str(p):sha256(p) for p in paths}
write_json(out/'prospective-freeze.json',dict(files=hashes,scope='Retrospective comparison; no new numerical acceptance gate'))
for name in names:
 r=json.loads((base/name/'report.json').read_text())
 assert r['sources_unchanged'] and r['all_cases_recorded']

def case(name,h,index=0):
 return json.loads((base/name/f'rebound-{h:.0e}-{index:02}.json').read_text())

summaries=[]
for h in [.0001,.0005]:
 before,before_shift,after,after_shift=[case(n,h) for n in names]
 def difference(a,b):
  assert len(a['rows'])==len(b['rows'])
  assert [r['time_s'] for r in a['rows']]==[r['time_s'] for r in b['rows']]
  amplitude=h*.001; speed=amplitude*a['omega_rad_s']
  return max(max(abs(x['displacement_m']-y['displacement_m'])/amplitude,abs(x['velocity_m_s']-y['velocity_m_s'])/speed) for x,y in zip(a['rows'],b['rows']))
 summaries.append(dict(side_m=h,timestep_s=2e-9,original_path_error=before['maximum_normalized_path_error'],original_shifted_path_error=before_shift['maximum_normalized_path_error'],local_path_error=after['maximum_normalized_path_error'],local_shifted_path_error=after_shift['maximum_normalized_path_error'],original_normalized_translation_difference=difference(before,before_shift),local_normalized_translation_difference=difference(after,after_shift),local_energy_error=after['maximum_relative_dynamic_energy_error']))

fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
styles=[('Original, centre 5 mm','tab:red','-'),('Original, centre 3h','tab:orange','--'),('Local, centre 5 mm','tab:blue','-'),('Local, centre 3h','tab:green','--')]
for name,(label,color,style) in zip(names,styles):
 rows=[case(name,.0001,i) for i in range(16)]
 axes[0].loglog([r['timestep_s']*1e9 for r in rows],[r['maximum_normalized_path_error'] for r in rows],style,color=color,label=label)
 r=rows[0];a=1e-7
 axes[1].semilogy([x['time_s']*1e6 for x in r['rows']],[max(abs(x['displacement_m']-x['exact_displacement_m'])/a,1e-12) for x in r['rows']],style,color=color,label=label)
axes[0].axhline(.01,ls=':',color='black',label='1% path criterion')
axes[0].set(xlabel='Timestep (ns)',ylabel='Maximum sampled normalized path error',title='0.1 mm cube rebound')
axes[1].set(xlabel='Physical time (µs)',ylabel='Absolute displacement error / initial depth',title='Same rebound at 2 ns')
for ax in axes:
 ax.grid(True,which='both',alpha=.2)
 ax.legend(fontsize=7)
fig.suptitle('Actual Metal: local plane distance and preserved initial-position residual',fontsize=11)
fig.tight_layout()
fig.savefig(out/'comparison.png',dpi=160)
unchanged=all(sha256(p)==v for p,v in hashes.items())
write_json(out/'report.json',dict(identity=out.name,inputs_unchanged=unchanged,cases=summaries,scope='Paired trajectory differences at identical observation times; synthetic plane contacts only. No assembly timestep change.'))
assert unchanged
print(json.dumps(summaries,indent=2))
