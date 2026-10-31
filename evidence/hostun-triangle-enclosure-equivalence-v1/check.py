import importlib.util
import json
import time
from pathlib import Path
import numpy as np
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.util import read_json,sha256,write_json

root=Path('evidence/hostun-triangle-enclosure-equivalence-v1')
p=read_json(root/'prospective-policy.json')
spec=importlib.util.spec_from_file_location('new_pruning',Path('examples/library_quadrature_culled.py'))
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
library=ShapeLibrary.load('assets/private/hostun-30-r64-s2048.h5')
shapes={s.id:s for s in library.shapes}
rows=[]
begin=time.perf_counter()
for item in p['cases']:
 assert sha256(item['path'])==item['sha256']
 expected=read_json(item['path']); case=expected['case']
 row=module.study_case(shapes[case['shape']],case,dict(subdivision_levels=list(range(len(expected['levels']))),query_batch_size=2048),root/'new-progress.json')
 for old,new in zip(expected['levels'],row['levels']):
  a=np.array(old['force_integral_m3']); b=np.array(new['force_integral_m3'])
  norm=np.linalg.norm(a)
  error=float(np.linalg.norm(a-b)/norm) if norm else float(np.linalg.norm(a-b))
  depth=abs(old['maximum_sampled_depth_fraction']-new['maximum_sampled_depth_fraction'])
  passed=error<=p['maximum_relative_force_difference'] and depth<=p['maximum_absolute_depth_difference'] and old['active_nodes']==new['active_nodes']
  rows.append(dict(shape=case['shape'],target=case['target_sampled_depth_fraction'],level=old['level'],relative_force_difference=error,absolute_depth_difference=depth,old_active_nodes=old['active_nodes'],new_active_nodes=new['active_nodes'],old_evaluated_nodes=old['evaluated_nodes'],new_evaluated_nodes=new['evaluated_nodes'],passed=passed))
 write_json(root/'progress.json',rows)
 print(case['shape'],case['target_sampled_depth_fraction'],len(row['levels']),flush=True)
write_json(root/'report.json',dict(policy=p,rows=rows,passed=all(r['passed'] for r in rows),elapsed_seconds=time.perf_counter()-begin))
raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
