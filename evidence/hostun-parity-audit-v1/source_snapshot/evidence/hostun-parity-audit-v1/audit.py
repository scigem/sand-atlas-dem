"""Observational diagnosis of ambiguous rays; no acceptance-gate replacement."""
import importlib.util
import hashlib
import numpy as np
import trimesh
from pathlib import Path
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.reference import triangle_nearest_points
from sand_atlas_dem.util import read_json,sha256,write_json

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
 return module

root=Path('evidence/hostun-parity-audit-v1')
p=read_json(root/'prospective-policy.json')
for path,digest in p['source_hashes'].items(): assert sha256(path)==digest
assert sha256(p['library'])==p['library_sha256']
assert sha256(p['pose_report'])==p['pose_report_sha256']
study=load('ray_culled_study','examples/library_quadrature_ray_cull.py')
winding=load('winding_diagnostic','examples/reference_winding.py')
shape=next(s for s in ShapeLibrary.load(p['library']).shapes if s.id==p['shape'])
unit=(6*shape.volume/np.pi)**(1/3)
mesh=trimesh.Trimesh(shape.vertices/unit,shape.faces,process=False)
ray=study.rays.BoundedRayIntersector(mesh)
planes=study.geometry.enclosure_planes(mesh.vertices)
bary=np.full((3,3),1/6)+np.eye(3)/2
rows=[]; summaries=[]
for case in read_json(p['pose_report'])['rows']:
 if case['shape']!=p['shape']: continue
 branch=np.array([case['separation_m']/unit,0,0])
 ambiguous=0
 for sign in [1,-1]:
  triangles=study.cull_source_triangles(mesh.triangles-sign*branch,mesh.bounds,planes)
  for _ in range(p['subdivisions']):
   triangles=study.cull_source_triangles(study.geometry.subdivide(triangles),mesh.bounds,planes)
  points=np.einsum('ij,fjk->fik',bary,triangles).reshape(-1,3)
  area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
  weights=np.repeat(area/3,3)
  for start in range(0,len(points),p['batch_size']):
   block=points[start:start+p['batch_size']]
   directions=np.tile(p['primary_ray_direction'],(len(block),1))
   _,indices,_=ray.intersects_location(np.vstack([block,block]),np.vstack([directions,-directions]),multiple_hits=True)
   hits=np.bincount(indices,minlength=2*len(block)).reshape(2,-1)
   broken=((hits[0]%2)!=(hits[1]%2)) & np.all(hits>0,axis=0)
   for local in np.flatnonzero(broken):
    ambiguous+=1
    if ambiguous>p['maximum_ambiguous_points_per_pose']:
     raise RuntimeError('Diagnostic point budget exhausted; partial audit retained')
    point=block[local]
    near,distance,_=triangle_nearest_points(mesh,point[None,:])
    w=winding.winding_number(mesh.triangles,point)
    old=[bool(mesh.contains(point[None,:])[0]) for _ in range(p['repeat_classifications'])]
    new=[bool(ray.contains_points(point[None,:])[0]) for _ in range(p['repeat_classifications'])]
    key=hashlib.sha256(point.tobytes()).hexdigest()
    np.savez(root/'private'/(key+'.npz'),point=point,nearest=near[0])
    row=dict(shape=shape.id,target_sampled_depth_fraction=case['target_sampled_depth_fraction'],pass_sign=sign,point_sha256=key,primary_forward_hits=int(hits[0,local]),primary_backward_hits=int(hits[1,local]),nearest_distance_diameters=float(distance[0]),winding_number=w,winding_integer_residual=float(abs(w-round(w))),unpruned_repeated_classifications=old,pruned_repeated_classifications=new,traction_integral_contribution_m3=(sign*.5*weights[start+local]*(near[0]-point)*unit**3))
    rows.append(row)
    write_json(root/'progress.json',dict(rows=rows,summaries=summaries))
  print(case['target_sampled_depth_fraction'],sign,'ambiguous',ambiguous,flush=True)
 summaries.append(dict(target_sampled_depth_fraction=case['target_sampled_depth_fraction'],ambiguous_points=ambiguous))
assert all(sha256(f)==h for f,h in p['source_hashes'].items())
write_json(root/'report.json',dict(policy=p,rows=rows,summaries=summaries,complete=True,scope='Observational primary-ray ambiguity and direct solid-angle diagnosis. No retrospective change to failed equivalence or scientific acceptance gates.'))
