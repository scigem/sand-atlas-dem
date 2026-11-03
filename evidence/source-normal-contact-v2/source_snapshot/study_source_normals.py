"""Frozen source-normal prototype diagnostics, independent static checks, Metal dynamics."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import trimesh

from sand_atlas_dem.annealing import select_timestep
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.gpu import Engine, RejectedStep
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import quat_matrix, sha256, write_json


def qmul(a, b):
    return np.r_[a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3],b[:3]),a[3]*b[3]-np.dot(a[:3],b[:3])]


def reference(library, spec, x, q, ids, scales, length):
    """Independent FP64 triangles, containment and source face normals; no dynamics."""
    force=np.zeros((2,3)); torque=np.zeros((2,3)); energy=0.
    rot=[quat_matrix(qi) for qi in q]
    meshes=[trimesh.Trimesh(library.shapes[int(i)].vertices/length,library.shapes[int(i)].faces,process=False) for i in ids]
    for src,dst in [(0,1),(1,0)]:
        shape=library.shapes[int(ids[src])]
        arm=shape.samples*scales[src] @ rot[src].T
        local=(x[src]+arm-x[dst]) @ rot[dst]/scales[dst]/length
        closest,distance,_=trimesh.proximity.closest_point(meshes[dst],local)
        inside=meshes[dst].contains(local)
        depth=distance*length*scales[dst]*inside
        if spec.contact_normal=='source-surface-experimental':
            _,_,face=trimesh.proximity.closest_point(meshes[src],shape.samples/length)
            normal=-meshes[src].face_normals[face] @ rot[src].T
        else:
            normal=np.divide(closest-local,distance[:,None],out=np.zeros_like(local),where=distance[:,None]>1e-14) @ rot[dst].T
        nodal=.5*spec.normal_stiffness*shape.areas[:,None]*scales[src]**2*depth[:,None]*normal
        force[src]+=nodal.sum(0);force[dst]-=nodal.sum(0)
        torque[src]+=np.cross(arm,nodal).sum(0)
        torque[dst]+=np.cross(x[src]+arm-x[dst],-nodal).sum(0)
        energy+=float(np.sum(.25*spec.normal_stiffness*shape.areas*scales[src]**2*depth**2))
    return force,torque,energy


def execute(out):
    p=json.loads((out/'protocol.json').read_text()); freeze=json.loads((out/'freeze.json').read_text())
    assert all(sha256(f)==h for f,h in freeze['inputs'].items())
    metal=None; statics=[]; pairs=[]; checks={}
    def build(library,spec,x,q,ids,scales,cell):
        nonlocal metal
        if shutil.disk_usage(out).free<p['minimum_free_bytes']: raise RuntimeError('Frozen disk floor reached')
        e=Engine(library,spec,x,cell,ids,scales,metal=metal)
        e.buffers['rotation'].copy_from_numpy(np.asarray(q,np.float32));e.initialize();metal=e.m
        return e
    def static(label,library,spec,x,q,ids,scales,cell):
        e=build(library,spec,x,q,ids,scales,cell)
        state=e.state();force=state['force'][:,:3]*e.force_unit;torque=state['torque'][:,:3]*e.energy_unit
        f,t,energy=reference(library,spec,x,q,ids,scales,e.length_unit)
        result={'label':label,'contact_normal':spec.contact_normal,'force_N':force.tolist(),'torque_Nm':torque.tolist(),
                'reference_force_N':f.tolist(),'reference_torque_Nm':t.tolist(),
                'force_error_N':float(np.linalg.norm(force-f)),'torque_error_Nm':float(np.linalg.norm(torque-t)),
                'force_check':bool(np.linalg.norm(force-f)<=p['force_atol_N']+p['force_rtol']*np.linalg.norm(f)),
                'torque_check':bool(np.linalg.norm(torque-t)<=p['torque_atol_Nm']+p['torque_rtol']*np.linalg.norm(t)),
                'reference_energy_proxy_J':energy,'diagnostics':e.diagnostics()}
        statics.append(result)
        print(label,spec.contact_normal,'force',force[1].tolist(),'reference checks',result['force_check'],result['torque_check'],flush=True)
        return e,result
    sphere=ShapeLibrary.load(p['sphere_fixture']);R=.0005
    base=Experiment(particles=2,contact_geometry='mesh',friction=0.,normal_stiffness=p['normal_stiffness'],
                    tangential_stiffness=.5*p['normal_stiffness'],timestep_policy='active-contact',timestep=1e-10,
                    max_overlap_fraction=.5,contacts_per_particle=16384)
    q=np.array([[0.,0.,0.,1.]]*2);cell=np.array([.02]*3);centre=cell*.5
    for separation in p['separations_over_R']:
        x=np.array([centre-[R*separation/2,0,0],centre+[R*separation/2,0,0]])
        for normal in ['target-gradient','source-surface-experimental']:
            e,_=static(f'sphere-{separation}',sphere,replace(base,contact_normal=normal),x,q,[0,0],[1.,1.],cell);del e
    exp=replace(base,contact_normal='source-surface-experimental')
    x=np.array([centre-[R*.75,0,0],centre+[R*.75,0,0]])
    e,original=static('objectivity-base',sphere,exp,x,q,[0,0],[1.,1.],cell);del e
    global_q=np.array([.2,-.3,.1,.9]);global_q/=np.linalg.norm(global_q);Q=quat_matrix(global_q)
    rotated_x=(x-centre) @ Q.T+centre;rotated_q=np.array([qmul(global_q,qi) for qi in q])
    e,rotated=static('objectivity-rotated',sphere,exp,rotated_x,rotated_q,[0,0],[1.,1.],cell);del e
    expected=np.array(original['force_N']) @ Q.T
    checks['global_rotation']=bool(np.linalg.norm(np.array(rotated['force_N'])-expected)<=p['objectivity_rtol']*np.linalg.norm(expected))
    e,swapped=static('body-swap',sphere,exp,x[::-1],q,[0,0],[1.,1.],cell);del e
    checks['body_swap']=bool(np.linalg.norm(np.array(swapped['force_N'])-np.array(original['force_N'])[::-1])<=p['objectivity_rtol']*np.linalg.norm(expected))
    contained_x=np.array([centre,centre+[.1*R,0,0]])
    e,_=static('unequal-contained',sphere,exp,contained_x,q,[0,0],[1.,.5],cell);del e
    try:
        e=build(sphere,exp,np.array([centre,centre]),q,[0,0],[1.,1.],cell)
        checks['coincidence_rejected']=False;del e
    except RejectedStep as exc:
        checks['coincidence_rejected']=bool(exc.flags & 16384)
    with h5py.File(p['pair_state']) as f:
        spec=Experiment(**json.loads(f.attrs['experiment']));library=ShapeLibrary.load(spec.shape_library)
    for bodies in p['pairs']:
        label='-'.join(map(str,bodies))
        with np.load(Path(p['pair_pose_dir'])/f'{label}-1024-control-final-pose.npz') as z:
            x=z['positions_m'];q=z['rotations_xyzw'];ids=z['shape_ids'];scales=z['scales'];cell=z['cell_m']
        variant=replace(spec,particles=2,contacts_per_particle=2048,contact_normal='source-surface-experimental')
        e,initial=static(label,library,variant,x,q,ids,scales,cell)
        meanD=np.mean([(6*library.shapes[int(i)].volume/np.pi)**(1/3)*s for i,s in zip(ids,scales)])
        h=p['fd_fraction']*meanD;angle=p['fd_fraction'];fdF=[];fdT=[]
        for axis in range(3):
            xp=x.copy();xm=x.copy();xp[1,axis]+=h;xm[1,axis]-=h
            fdF.append(-(reference(library,variant,xp,q,ids,scales,e.length_unit)[2]-reference(library,variant,xm,q,ids,scales,e.length_unit)[2])/(2*h))
            dq=np.zeros(4);dq[axis]=np.sin(angle/2);dq[3]=np.cos(angle/2)
            qp=q.copy();qm=q.copy();qp[1]=qmul(dq,q[1]);dq[:3]*=-1;qm[1]=qmul(dq,q[1])
            fdT.append(-(reference(library,variant,x,qp,ids,scales,e.length_unit)[2]-reference(library,variant,x,qm,ids,scales,e.length_unit)[2])/(2*angle))
        force=np.array(initial['reference_force_N'])[1];torque=np.array(initial['reference_torque_Nm'])[1]
        errors={'translation':float(np.linalg.norm(force-fdF)/max(np.linalg.norm(force),np.linalg.norm(fdF),1e-30)),
                'rotation':float(np.linalg.norm(torque-fdT)/max(np.linalg.norm(torque),np.linalg.norm(fdT),1e-30))}
        rows=[e.diagnostics()];elapsed=0.;rejections=[];failure=None
        try:
            while e.step<p['steps']:
                limits=e.timestep_limits();dt=select_timestep(limits['elastic_seconds'],limits['viscous_seconds'],p['timestep_cap_s'])
                amount=min(p['batch_steps'],p['steps']-e.step)
                for retry in range(9):
                    if dt<p['minimum_timestep_s']:raise RuntimeError('Frozen dt floor reached')
                    e.spec=replace(e.spec,timestep=dt)
                    try:e.advance(amount,drag=p['drag_per_step']/dt);break
                    except RejectedStep as exc:
                        rejections.append({'step':e.step,'dt':dt,'flags':exc.flags})
                        if exc.flags & ~(8192|4) or retry==8:raise
                        dt*=.5
                elapsed+=amount*dt
                if e.step%p['diagnostic_steps']==0:
                    row=e.diagnostics();row['explicit_elapsed_time_s']=elapsed;rows.append(row)
        except Exception as exc:failure=f'{type(exc).__name__}: {exc}'
        state=e.state();np.savez_compressed(out/f'{label}-final-state.npz',**state)
        np.savez_compressed(out/f'{label}-final-pose.npz',positions_m=(state['position'][:,:3].astype(float)+state['positionRemainder'][:,:3])*e.length_unit,
                            rotations_xyzw=state['rotation'].astype(float)+state['rotationRemainder'],shape_ids=ids,scales=scales,cell_m=cell)
        result={'bodies':bodies,'initial':initial,'gradient_errors':errors,'negative_energy_gradient_force_N':fdF,
                'negative_energy_gradient_torque_Nm':fdT,'history':rows,'final':e.diagnostics(),'steps':e.step,
                'failure':failure,'rejections':rejections,'length_unit_m':e.length_unit,'time_unit_s':e.time_unit}
        pairs.append(result);write_json(out/'latest.json',{'statics':statics,'pairs':pairs,'checks':checks})
        print('Pair',bodies,'final overlap',result['final']['max_overlap_fraction'],'gradient errors',errors,'failure',failure,flush=True)
        del e
    checks['independent_force_torque']=all(r['force_check'] and r['torque_check'] for r in statics)
    checks['sphere_repulsive']=all(r['force_N'][1][0]>0 for r in statics if r['label'].startswith('sphere-') and r['contact_normal']==exp.contact_normal)
    target=next(r for r in statics if r['label']=='sphere-1.99' and r['contact_normal']=='target-gradient')
    source=next(r for r in statics if r['label']=='sphere-1.99' and r['contact_normal']==exp.contact_normal)
    checks['shallow_force_agreement']=bool(abs(source['force_N'][1][0]/target['force_N'][1][0]-1)<=p['shallow_force_rtol'])
    checks['execution']=all(r['failure'] is None for r in pairs)
    checks['input_integrity']=all(sha256(f)==h for f,h in freeze['inputs'].items())
    checks['source_integrity']=all(sha256(out/'source_snapshot'/f)==h for f,h in freeze['sources'].items())
    physical={'existing_proxy_gradient_consistency':all(max(r['gradient_errors'].values())<=p['physical_gradient_rtol'] for r in pairs),
              'pair_clearance_below_original_one_percent':all(r['final']['max_overlap_fraction']<=.01 for r in pairs)}
    write_json(out/'report.json',{'identity':p['identity'],'software_checks':checks,'software_passed':all(checks.values()),
                                'physical_checks':physical,'physical_passed':all(physical.values()),'scientific_admission':False,
                                'statics':statics,'pairs':pairs,'scope':p['scope']})
    return 0 if all(checks.values()) else 1


def main():
    ap=argparse.ArgumentParser();ap.add_argument('protocol',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--worker',action='store_true');a=ap.parse_args();out=a.output.resolve()
    if a.worker:return execute(out)
    root=Path(__file__).resolve().parents[1];out.mkdir(parents=True,exist_ok=False);snap=out/'source_snapshot'
    shutil.copytree(root/'src/sand_atlas_dem',snap/'sand_atlas_dem',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,snap/Path(__file__).name);shutil.copy2(a.protocol,out/'protocol.json');p=json.loads(a.protocol.read_text())
    with h5py.File(p['pair_state']) as f:shape=json.loads(f.attrs['experiment'])['shape_library'];assert sha256(shape)==f.attrs['shape_sha256']
    inputs=[a.protocol,p['sphere_fixture'],p['pair_state'],shape]+[Path(p['pair_pose_dir'])/('-'.join(map(str,pair))+'-1024-control-final-pose.npz') for pair in p['pairs']]
    write_json(out/'freeze.json',{'inputs':{str(Path(f).resolve()):sha256(f) for f in inputs},'sources':{str(f.relative_to(snap)):sha256(f) for f in snap.rglob('*') if f.is_file()},
                                'post_study_regression':'Full existing pytest suite, explicit Metal, serial execution; all failures retained, no gate changes.'})
    with (out/'run.log').open('w') as log:
        r=subprocess.run([sys.executable,str(snap/Path(__file__).name),str(out/'protocol.json'),'--output',str(out),'--worker'],cwd=root,
                         env=dict(os.environ,PYTHONPATH=str(snap),SAND_DEM_METAL='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT,check=False)
    write_json(out/'execution.json',{'returncode':r.returncode});return r.returncode


if __name__=='__main__':raise SystemExit(main())
