"""Frozen static audit of a failed annealing state; no physical-time simulation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import trimesh

from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import quat_matrix, sha256, write_json


def execute(out):
    protocol = json.loads((out/'protocol.json').read_text())
    freeze = json.loads((out/'freeze.json').read_text())
    assert all(sha256(path)==digest for path,digest in freeze['inputs'].items())
    assert all(sha256(out/'source_snapshot'/name)==digest for name,digest in freeze['sources'].items())
    with h5py.File(protocol['state']) as f:
        spec=Experiment(**json.loads(f.attrs['experiment']))
        saved={k:f[k][:] for k in f if k not in {'shape_ids','scales'}}
        saved['contacts']=int(f.attrs['contacts'])
        ids=f['shape_ids'][:]
        scales=f['scales'][:]
        step=int(f.attrs['step'])
        length=float(f.attrs['length_unit'])
    library=ShapeLibrary.load(spec.shape_library)
    pos=(saved['position'][:,:3].astype(float)+saved['positionRemainder'][:,:3])*length
    cell=saved['cell'][0,:3].astype(float)*length
    engine=Engine(library,spec,pos,cell,ids,scales)
    engine.restore(saved,step)
    state=engine.state()
    count=state['contacts']
    keys=state['meta'][:,:3].astype(int)
    pd=engine.m.read_prefix(engine.buffers['pointDepth'],4*count,shape=(count,4)).astype(float)
    na=state['normalArea'].astype(float)
    force=engine.m.read_prefix(engine.buffers['contactForce'],4*count,shape=(count,4))[:,:3].astype(float)*engine.force_unit
    depth=pd[:,3]*length
    diam=engine.body[:,3].astype(float)*length
    ratios=depth/np.minimum(diam[keys[:,0]],diam[keys[:,1]])
    traction=.5*spec.normal_stiffness*(na[:,3]*length**2*depth)[:,None]*na[:,:3]
    traction_error=float(np.max(np.linalg.norm(force-traction,axis=1)))
    summed=np.zeros((len(ids),3));torque=np.zeros_like(summed)
    arms=pd[:,:3]*length
    branch=pos[keys[:,1]]-pos[keys[:,0]]
    branch-=np.round(branch/cell)*cell
    np.add.at(summed,keys[:,0],force)
    np.add.at(summed,keys[:,1],-force)
    np.add.at(torque,keys[:,0],np.cross(arms,force))
    np.add.at(torque,keys[:,1],np.cross(arms-branch,-force))
    actual=state['force'][:,:3].astype(float)*engine.force_unit
    actual_t=state['torque'][:,:3].astype(float)*engine.force_unit*length
    fscale=float(np.max(np.linalg.norm(summed,axis=1)))
    tscale=float(np.max(np.linalg.norm(torque,axis=1)))
    gather_f_error=float(np.max(np.linalg.norm(actual-summed,axis=1)))
    gather_t_error=float(np.max(np.linalg.norm(actual_t-torque,axis=1)))
    rotations=[]
    for q in state['rotation'].astype(float)+state['rotationRemainder']:
        rotations.append(quat_matrix(q/np.linalg.norm(q)))
    meshes={}
    def mesh(index):
        if index not in meshes:
            s=library.shapes[index]
            meshes[index]=trimesh.Trimesh(vertices=s.vertices/length,faces=s.faces,process=False)
        return meshes[index]
    def query(src,dst,source_nodes,positions):
        source=library.shapes[int(ids[src])]
        arm=source_nodes*scales[src] @ rotations[src].T
        delta=positions[dst]-positions[src]
        delta-=np.round(delta/cell)*cell
        local=(arm-delta) @ rotations[dst]/scales[dst]/length
        m=mesh(int(ids[dst]))
        closest,distance,tri=trimesh.proximity.closest_point(m,local)
        inside=m.contains(local)
        normal=np.divide(closest-local,distance[:,None],out=np.zeros_like(local),where=distance[:,None]>1e-14)
        normal=normal @ rotations[dst].T
        return inside, distance*length*scales[dst], normal
    # Independently query the deepest captured nodes, grouped by ordered pair.
    selected=np.argsort(ratios)[-protocol['deepest_nodes']:]
    query_rows=[]
    for src,dst in np.unique(keys[selected,:2],axis=0):
        subset=selected[(keys[selected,0]==src)&(keys[selected,1]==dst)]
        nodes=library.shapes[int(ids[src])].samples[keys[subset,2]]
        inside,d,normals=query(int(src),int(dst),nodes,pos)
        for index,is_inside,distance,normal in zip(subset,inside,d,normals):
            query_rows.append({'contact':int(index),'source':int(src),'target':int(dst),'node':int(keys[index,2]),
                'shape_source':library.shapes[int(ids[src])].id,'shape_target':library.shapes[int(ids[dst])].id,
                'gpu_overlap':float(ratios[index]),'depth_m':float(depth[index]),'diameter_denominator_m':float(min(diam[src],diam[dst])),
                'cpu_inside':bool(is_inside),'depth_error_m':float(abs(distance-depth[index])),
                'normal_error':float(np.linalg.norm(normal-na[index,:3]))})
    # Full independent1024-node surface integration for the three worst pairs.
    chosen=[]
    for index in np.argsort(ratios)[::-1]:
        pair=tuple(sorted(keys[index,:2]))
        if pair not in chosen:
            chosen.append(pair)
        if len(chosen)==protocol['pair_count']:
            break
    pair_rows=[]
    def pair_reference(a,b,positions):
        f=np.zeros(3);energy=0.;deep=0.;counts=0
        for src,dst in [(a,b),(b,a)]:
            s=library.shapes[int(ids[src])]
            inside,d,normals=query(src,dst,s.samples,positions)
            magnitude=.5*spec.normal_stiffness*s.areas*scales[src]**2*d*inside
            f+=(1 if src==b else -1)*np.sum(magnitude[:,None]*normals,axis=0)
            energy+=float(np.sum(.25*spec.normal_stiffness*s.areas*scales[src]**2*d**2*inside))
            deep=max(deep,float(np.max(d*inside)/min(diam[a],diam[b])))
            counts+=int(inside.sum())
        return f,energy,deep,counts
    for a,b in chosen:
        mask=((keys[:,0]==a)&(keys[:,1]==b))|((keys[:,0]==b)&(keys[:,1]==a))
        gpu=np.sum(np.where((keys[mask,0]==b)[:,None],1.,-1.)*force[mask],axis=0)
        reference,energy,worst,cpu_count=pair_reference(a,b,pos)
        centre=pos[b]-pos[a];centre-=np.round(centre/cell)*cell
        distance=np.linalg.norm(centre)
        direction=centre/distance
        pair_rows.append({'bodies':[int(a),int(b)],'shapes':[library.shapes[int(ids[i])].id for i in [a,b]],
            'centre_distance_over_mean_diameter':float(distance/np.mean(diam[[a,b]])),
            'max_overlap_independent':worst,'gpu_contact_count':int(mask.sum()),'cpu_contact_count':cpu_count,
            'force_on_second_gpu_N':gpu.tolist(),'force_on_second_cpu_N':reference.tolist(),
            'force_relative_error':float(np.linalg.norm(gpu-reference)/max(np.linalg.norm(reference),1e-30)),
            'radial_separating_force_N':float(np.dot(reference,direction)),
            'nodal_force_cancellation_ratio':float(np.linalg.norm(gpu)/max(np.linalg.norm(force[mask],axis=1).sum(),1e-30)),
            'pair_energy_proxy_J':energy})
    max_distance=max(r['depth_error_m'] for r in query_rows)
    max_normal=max(r['normal_error'] for r in query_rows)
    checks={
      'deepest_nodes_inside':all(r['cpu_inside'] for r in query_rows),
      'distance_reference':max_distance<=protocol['distance_atol_m'],
      'normal_reference':max_normal<=protocol['normal_atol'],
      'normal_traction':traction_error<=protocol['force_atol_N']+protocol['force_rtol']*float(np.max(np.linalg.norm(traction,axis=1))),
      'gather_force':gather_f_error<=protocol['force_atol_N']+protocol['force_rtol']*fscale,
      'gather_torque':gather_t_error<=protocol['torque_atol_Nm']+protocol['force_rtol']*tscale,
      'pair_reference':all(r['force_relative_error']<=protocol['pair_force_rtol'] and r['cpu_contact_count']==r['gpu_contact_count'] for r in pair_rows),
      'input_integrity':all(sha256(path)==digest for path,digest in freeze['inputs'].items()),
      'source_integrity':all(sha256(out/'source_snapshot'/name)==digest for name,digest in freeze['sources'].items()),
    }
    report={'identity':protocol['identity'],'checks':checks,'passed':all(checks.values()),
        'captured_contacts':count,'maximum_overlap':float(ratios.max()),'minimum_equivalent_diameter_m':float(diam.min()),
        'maximum_equivalent_diameter_m':float(diam.max()),'traction_error_N':traction_error,
        'gather_force_error_N':gather_f_error,'gather_torque_error_Nm':gather_t_error,
        'max_cpu_distance_error_m':max_distance,'max_cpu_normal_error':max_normal,
        'deepest_nodes':sorted(query_rows,key=lambda r:r['gpu_overlap'],reverse=True),'pairs':pair_rows,
        'scope':'Static diagnostic on retained coarse construction state. CPU ray containment and triangle queries are independent of Slang BVH. No simulation, physical-law admission, or full-quadrature qualification.'}
    write_json(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['deepest_nodes']},indent=2),flush=True)
    return 0 if report['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker',action='store_true')
    a=parser.parse_args();out=a.output.resolve()
    if a.worker:
        return execute(out)
    root=Path(__file__).resolve().parents[1];out.mkdir(exist_ok=False,parents=True);snap=out/'source_snapshot'
    shutil.copytree(root/'src/sand_atlas_dem',snap/'sand_atlas_dem',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,snap/Path(__file__).name);shutil.copy2(a.protocol,out/'protocol.json')
    p=json.loads(a.protocol.read_text())
    with h5py.File(p['state']) as f:
        shape=json.loads(f.attrs['experiment'])['shape_library']
        assert sha256(shape)==f.attrs['shape_sha256']
    inputs={str(Path(path).resolve()):sha256(path) for path in [a.protocol,p['state'],shape]}
    write_json(out/'freeze.json',{'inputs':inputs,'sources':{str(path.relative_to(snap)):sha256(path) for path in snap.rglob('*') if path.is_file()}})
    env=dict(os.environ,PYTHONPATH=str(snap),SAND_DEM_METAL='1',PYTHONUNBUFFERED='1')
    with (out/'run.log').open('w') as log:
        r=subprocess.run([sys.executable,str(snap/Path(__file__).name),str(out/'protocol.json'),'--output',str(out),'--worker'],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=False)
    write_json(out/'execution.json',{'returncode':r.returncode})
    return r.returncode


if __name__=='__main__':
    raise SystemExit(main())
