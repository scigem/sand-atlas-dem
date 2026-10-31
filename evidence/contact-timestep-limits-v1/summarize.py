from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/private/tmp/sand-atlas-dem-walls/evidence/contact-timestep-limits-v1')
r=json.loads((root/'report.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(11,4.3))
summary=[]
for mode,ax in zip(['confined','rebound'],axs):
 for h in sorted(set(x['side_m'] for x in r['cases'])):
  rows=sorted([x for x in r['cases'] if x['mode']==mode and x['side_m']==h],key=lambda x:x['timestep_s'])
  ax.loglog([x['timestep_s']*1e9 for x in rows],[max(x['maximum_normalized_path_error'],1e-8) for x in rows],'.-',label=f'{h*1e3:g} mm')
  good=[x for x in rows if x['accurate']]
  summary.append(dict(mode=mode,side_m=h,production_bound_ns=rows[0]['production_bound_s']*1e9,largest_tested_accurate_ns=max((x['timestep_s']*1e9 for x in good),default=None),linear_verlet_limit_ns=rows[0]['linear_verlet_stability_threshold_s']*1e9,failed_cases=sum(x['failure'] is not None for x in rows),accurate_cases=len(good)))
 ax.axhline(.01,color='black',ls='--',label='1% path criterion')
 ax.set(title=mode.capitalize(),xlabel='Timestep (ns)',ylabel='Maximum sampled normalized path error')
 ax.grid(True,which='both',alpha=.25)
 ax.legend(fontsize=8)
fig.suptitle('Isolated analytical contact fixtures — actual Metal\nAccuracy also requires ≤1% dynamic-energy error; no assembly qualification',fontsize=11)
fig.tight_layout()
fig.savefig(root/'timestep-error.png',dpi=160)
(root/'summary.json').write_text(json.dumps(dict(cases=summary,sources_unchanged=r['sources_unchanged'],all_cases_recorded=r['all_cases_recorded']),indent=2)+'\n')
print(json.dumps(summary,indent=2))
