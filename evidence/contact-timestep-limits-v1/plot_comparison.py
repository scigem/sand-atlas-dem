"""Retrospective plots; all numerical results and original plot remain unchanged."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
parent=json.loads((root/'report.json').read_text())
origin=json.loads((root.parent/'contact-timestep-origin-diagnostic-v1/report.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for mode,ax in zip(['confined','rebound'],axes):
 for h in sorted(set(r['side_m'] for r in parent['cases'])):
  rows=sorted([r for r in parent['cases'] if r['mode']==mode and r['side_m']==h],key=lambda r:r['timestep_s'])
  x=[r['timestep_s']*1e9 for r in rows]
  y=[r['maximum_normalized_path_error'] if r['completed'] and not r['analytical_domain_exit'] else np.nan for r in rows]
  line,=ax.loglog(x,y,'.-',label=f'{h*1e3:g} mm; centre 5 mm')
  failed=[r for r in rows if not r['completed'] or r['analytical_domain_exit']]
  if failed:
   ax.scatter([r['timestep_s']*1e9 for r in failed],[10]*len(failed),marker='x',color=line.get_color())
  if mode=='rebound':
   shifted=sorted([r for r in origin['cases'] if r['side_m']==h],key=lambda r:r['timestep_s'])
   if shifted:
    ax.loglog([r['timestep_s']*1e9 for r in shifted],[r['maximum_normalized_path_error'] for r in shifted],'--',color=line.get_color(),label=f'{h*1e3:g} mm; centre 3h')
 ax.axhline(.01,color='black',ls=':',label='1% path criterion')
 ax.set(title=mode.capitalize(),xlabel='Timestep (ns)',ylabel='Maximum sampled normalized path error')
 ax.grid(True,which='both',alpha=.2)
 ax.legend(fontsize=7)
fig.suptitle('Actual Metal: isolated contacts against exact solutions\nCrosses at 10 mark stopped / rejected / domain-exit cases, not measured error; energy criterion also applies',fontsize=10)
fig.tight_layout()
fig.savefig(root/'timestep-comparison.png',dpi=160)
