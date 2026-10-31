import importlib.util
from pathlib import Path
import numpy as np
import trimesh
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.util import read_json,sha256,write_json

root=Path('evidence/hostun-deterministic-inside-v1')
p=read_json(root/'prospective-policy.json')
assert sha256(p['audit_report'])==p['audit_report_sha256']
assert sha256(p['library'])==p['library_sha256']
assert all(sha256(f)==h for f,h in p['source_hashes'].items())
audit=read_json(p['audit_report'])
shape=next(s for s in ShapeLibrary.load(p['library']).shapes if s.id==audit['policy']['shape'])
unit=(6*shape.volume/np.pi)**(1/3)
mesh=trimesh.Trimesh(shape.vertices/unit,shape.faces,process=False)
points=np.array([np.load(Path(p['audit_report']).parent/'private'/(r['point_sha256']+'.npz'))['point'] for r in audit['rows']])
expected=np.array([round(r['winding_number'])==1 for r in audit['rows']])
spec=importlib.util.spec_from_file_location('deterministic_inside','examples/reference_inside.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
query=module.DeterministicRayIntersector(mesh,p['winding_tolerance'])
checks=[]
for repeat in range(p['repetitions']):
 for batch in p['batch_sizes']:
  for reverse in [False,True]:
   ordered=points[::-1] if reverse else points
   actual=np.concatenate([query.contains_points(ordered[i:i+batch]) for i in range(0,len(ordered),batch)])
   if reverse: actual=actual[::-1]
   checks.append(dict(repetition=repeat,batch=batch,reversed=reverse,passed=bool(np.array_equal(actual,expected))))
assert all(sha256(f)==h for f,h in p['source_hashes'].items())
write_json(root/'report.json',dict(policy=p,checks=checks,point_count=len(points),winding_checks=query.winding_checks,maximum_integer_residual=query.maximum_integer_residual,passed=all(c['passed'] for c in checks)))
print('passed',all(c['passed'] for c in checks),'checks',len(checks),'points',len(points),'winding_evaluations',query.winding_checks,flush=True)
