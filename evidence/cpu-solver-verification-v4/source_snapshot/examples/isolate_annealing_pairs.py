"""Static energy-gradient check and isolated actual-Metal pair relaxation."""
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


def execute(out):
    p=json.loads((out/'protocol.json').read_text());freeze=json.loads((out/'freeze.json').read_text())
    assert all(sha256(path)==h for path,h in freeze['inputs'].items())
    with h5py.File(p['state']) as f:
        spec=Experiment(**json.loads(f.attrs['experiment']))
        length=float(f.attrs['length_unit']);ids=f['shape_ids'][:];scales=f['scales'][:]
        pos=(f['position'][:,:3].astype(float)+f['positionRemainder'][:,:3])*length
        cell=f['cell'][0,:3].astype(float)*length
        q=f['rotation'][:].astype(float)+f['rotationRemainder'][:]
        q/=np.linalg.norm(q,axis=1)[:,None]
    library=ShapeLibrary.load(spec.shape_library)
    diam=np.array([(6*library.shapes[int(i)].volume/np.pi)**(1/3)*s for i,s in zip(ids,scales)])
    pairs=json.loads(Path(p['audit_report']).read_text())['pairs']
    results=[];metal=None
    for pair in pairs:
        a,b=pair['bodies'];body=[a,b]
        branch=pos[b]-pos[a];branch-=np.round(branch/cell)*cell
        positions=np.array([cell*.5-branch*.5,cell*.5+branch*.5])
        rotations=[quat_matrix(q[i]) for i in body]
        meshes=[trimesh.Trimesh(library.shapes[int(ids[i])].vertices/length,library.shapes[int(ids[i])].faces,process=False) for i in body]
        def reference(coordinates):
            force=np.zeros(3);energy=0.
            delta=coordinates[1]-coordinates[0]
            for src,dst in [(0,1),(1,0)]:
                s=library.shapes[int(ids[body[src]])]
                arm=s.samples*scales[body[src]] @ rotations[src].T
                local=(arm-(delta if src==0 else -delta)) @ rotations[dst]/scales[body[dst]]/length
                closest,distance,_=trimesh.proximity.closest_point(meshes[dst],local)
                inside=meshes[dst].contains(local)
                normal=np.divide(closest-local,distance[:,None],out=np.zeros_like(local),where=distance[:,None]>1e-14) @ rotations[dst].T
                depth=distance*length*scales[body[dst]]
                magnitude=.5*spec.normal_stiffness*s.areas*scales[body[src]]**2*depth*inside
                force+=(1 if src==1 else -1)*np.sum(magnitude[:,None]*normal,axis=0)
                energy+=float(np.sum(.25*spec.normal_stiffness*s.areas*scales[body[src]]**2*depth**2*inside))
            return force,energy
        f,energy=reference(positions)
        derivatives=[]
        for fraction in p['difference_steps_over_diameter']:
            h=fraction*np.mean(diam[body]);fd=[]
            for axis in range(3):
                plus=positions.copy();minus=positions.copy()
                plus[1,axis]+=h;minus[1,axis]-=h
                fd.append(-(reference(plus)[1]-reference(minus)[1])/(2*h))
            derivatives.append({'h_over_mean_diameter':fraction,'force_N':fd,
                'relative_error':float(np.linalg.norm(np.asarray(fd)-f)/max(np.linalg.norm(f),1e-30))})
        engine=Engine(library,replace(spec,particles=2,contacts_per_particle=2048),positions,cell,ids[body],scales[body],metal=metal)
        engine.buffers['rotation'].copy_from_numpy(q[body].astype(np.float32));engine.initialize();metal=engine.m
        rows=[engine.diagnostics()];rejections=[];failure=None;elapsed=0.
        try:
            while engine.step<p['steps']:
                limits=engine.timestep_limits()
                dt=select_timestep(limits['elastic_seconds'],limits['viscous_seconds'],p['timestep_cap_s'])
                amount=min(p['batch_steps'],p['steps']-engine.step)
                for retry in range(9):
                    if dt<p['minimum_timestep_s']:
                        raise RuntimeError('Timestep below fixed minimum')
                    engine.spec=replace(engine.spec,timestep=dt)
                    try:
                        engine.advance(amount,drag=p['drag_per_step']/dt)
                        break
                    except RejectedStep as exc:
                        rejections.append({'step':engine.step,'dt':dt,'flags':exc.flags})
                        if exc.flags & ~(8192|4) or retry==8:
                            raise
                        dt*=.5
                elapsed+=amount*dt
                if engine.step%p['diagnostic_steps']==0:
                    row=engine.diagnostics();row['explicit_elapsed_time_s']=elapsed;rows.append(row)
        except Exception as exc:
            failure=f'{type(exc).__name__}: {exc}'
        final=engine.diagnostics()
        results.append({'bodies':body,'reference_force_N':f.tolist(),'reference_energy_J':energy,
            'energy_derivatives':derivatives,'initial_overlap':rows[0]['max_overlap_fraction'],
            'final_overlap':final['max_overlap_fraction'],'final_pairs':final['active_grain_pairs'],
            'steps':engine.step,'failure':failure,'rejections':rejections,'history':rows})
        write_json(out/'latest.json',{'completed_pairs':len(results),'results':results})
        print('Pair',body,'overlap',results[-1]['initial_overlap'],'->',results[-1]['final_overlap'],'steps',engine.step,'failure',failure,flush=True)
        del engine
    checks={'energy_gradient':all(r['energy_derivatives'][-1]['relative_error']<=p['gradient_rtol'] for r in results),
        'input_integrity':all(sha256(path)==h for path,h in freeze['inputs'].items()),
        'source_integrity':all(sha256(out/'source_snapshot'/name)==h for name,h in freeze['sources'].items())}
    write_json(out/'report.json',{'identity':p['identity'],'checks':checks,'passed':all(checks.values()),'results':results,
        'scope':'Gradient check at retained poses and isolated-pair dynamics diagnostic; overlap clearance is observed, not an admission gate. No dense specimen prepared.'})
    return 0 if all(checks.values()) else 1


def main():
    ap=argparse.ArgumentParser();ap.add_argument('protocol',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--worker',action='store_true');a=ap.parse_args();out=a.output.resolve()
    if a.worker:return execute(out)
    root=Path(__file__).resolve().parents[1];out.mkdir(exist_ok=False,parents=True);snap=out/'source_snapshot'
    shutil.copytree(root/'src/sand_atlas_dem',snap/'sand_atlas_dem',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,snap/Path(__file__).name);shutil.copy2(a.protocol,out/'protocol.json');p=json.loads(a.protocol.read_text())
    with h5py.File(p['state']) as f:shape=json.loads(f.attrs['experiment'])['shape_library'];assert sha256(shape)==f.attrs['shape_sha256']
    write_json(out/'freeze.json',{'inputs':{str(Path(path).resolve()):sha256(path) for path in [a.protocol,p['state'],p['audit_report'],shape]},'sources':{str(path.relative_to(snap)):sha256(path) for path in snap.rglob('*') if path.is_file()}})
    with (out/'run.log').open('w') as log:
        r=subprocess.run([sys.executable,str(snap/Path(__file__).name),str(out/'protocol.json'),'--output',str(out),'--worker'],cwd=root,env=dict(os.environ,PYTHONPATH=str(snap),SAND_DEM_METAL='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT,check=False)
    write_json(out/'execution.json',{'returncode':r.returncode});return r.returncode


if __name__=='__main__':raise SystemExit(main())
