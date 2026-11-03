"""Queue a frozen, short Metal triaxial pilot after accepted preparation."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np

from sand_atlas_dem.annealing import select_timestep
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.gpu import Engine, RejectedStep
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import sha256, write_json
from packing_pressure_control import pressure_rates
from study_annealed_preparation import snapshot


def preparation_ready(report, rows, p):
    if not report.get('passed') or report.get('failure') or not report.get('source_integrity') or not report.get('input_integrity'):
        return False
    count = p['converged_samples']
    if len(rows) < count:
        return False
    return all(max(abs(r['stress'][i][i]/p['target_pressure_Pa']-1) for i in range(3)) <= p['pressure_tolerance']
               and r['kinetic_energy']/max(r['elastic_energy_proxy'], 1e-30) <= p['kinetic_tolerance']
               and r['unbalanced_force'] <= p['force_tolerance']
               and r['max_overlap_fraction'] <= p['equilibrium_overlap_limit'] for r in rows[-count:])


def loading_rates(row, p, dt, count, axial_rate):
    rates, _ = pressure_rates(row['stress'], row['max_overlap_fraction'], p, count*dt)
    rates[2] = -axial_rate
    if abs(rates[2])*count*dt > p['max_batch_log_strain']:
        raise RuntimeError('Axial strain exceeds frozen batch bound')
    return rates


def restart_checks(engine, state, p):
    engine.restore(state, p['start_step'])
    for k in ['accumulated', 'accumulatedRemainder']:
        engine.buffers[k].copy_from_numpy(state[k])
    actual = engine.state()
    ia = np.lexsort(actual['meta'][:, :3].T[::-1]); ib = np.lexsort(state['meta'][:, :3].T[::-1])
    keys = np.array_equal(actual['meta'][ia], state['meta'][ib])
    return dict(primary_exact=all(np.array_equal(actual[k], state[k]) for k in p['restart_primary_exact']),
        contact_keys_exact=keys, contact_count_exact=actual['contacts']==state['contacts'],
        keyed_normals_exact=keys and np.array_equal(actual['normalArea'][ia], state['normalArea'][ib]),
        keyed_slip_bound=keys and np.allclose(actual['slip'][ia], state['slip'][ib], rtol=p['restart_slip_rtol'], atol=p['restart_slip_atol']),
        resultants_bound=all(np.allclose(actual[k], state[k], rtol=p['restart_resultant_rtol'], atol=p['restart_resultant_atol']) for k in ['force','torque','accumulatedRemainder']))


def execute(out, p):
    parent = Path(p['preparation_run'])
    report = json.loads((parent/'report.json').read_text())
    rows = [json.loads(s) for s in (parent/'telemetry.jsonl').read_text().splitlines() if s.strip()]
    if not preparation_ready(report, rows, p):
        raise RuntimeError('Preparation has not passed independent final-window audit')
    checkpoint = parent/'state-final.h5'
    with h5py.File(checkpoint) as f:
        spec = Experiment(**json.loads(f.attrs['experiment']))
        if sha256(spec.shape_library) != f.attrs['shape_sha256']:
            raise RuntimeError('Shape identity mismatch')
        state = {k:f[k][:] for k in f if k not in ('shape_ids','scales')}; state['contacts']=int(f.attrs['contacts'])
        ids, scales = f['shape_ids'][:], f['scales'][:]
        p['start_step'] = int(f.attrs['step'])
        length, time_unit = float(f.attrs['length_unit']), float(f.attrs['time_unit'])
    if p['start_step'] != rows[-1]['step'] or p['start_step'] != report['total_physical_step']:
        raise RuntimeError('Preparation final checkpoint/report/telemetry step mismatch')
    inputs = [checkpoint,parent/'report.json',parent/'telemetry.jsonl',Path(spec.shape_library)]
    write_json(out/'execution-freeze.json', {'inputs':{str(f.resolve()):sha256(f) for f in inputs},
        'note':'Future preparation artifacts bound after final acceptance and before any pilot GPU execution'})
    library = ShapeLibrary.load(spec.shape_library)
    x = (state['position'][:, :3].astype(float)+state['positionRemainder'][:, :3])*length
    cell = state['cell'][0, :3].astype(float)*length
    origin_log = state['cellLog'][0, :3].astype(float)+state['cellLog'][1, :3]
    diameters = np.array([(6*library.shapes[int(i)].volume/np.pi)**(1/3)*s for i,s in zip(ids,scales)])
    metal = None; results = []
    for case in p['cases']:
        target = out/case['id']; target.mkdir()
        engine = Engine(library, spec, x, cell, ids, scales, metal=metal)
        if engine.length_unit != length or engine.time_unit != time_unit:
            raise RuntimeError('Restart units mismatch')
        checks = restart_checks(engine, state, p); write_json(target/'restart-checks.json',checks)
        if not all(checks.values()):
            raise RuntimeError('Pilot restart checks failed')
        metal = engine.m
        initial = engine.diagnostics(); write_json(target/'initial.json', initial)
        elapsed = 0.; strain = np.zeros(3); rejected=[]; failure=None; begun=time.monotonic(); records=[]
        try:
            with (target/'telemetry.jsonl').open('w') as log:
                while strain[2] < p['axial_engineering_strain']-p['terminal_strain_tolerance']:
                    if shutil.disk_usage(out).free < p['minimum_free_bytes']:
                        raise RuntimeError('Frozen disk reserve reached')
                    if engine.step-p['start_step'] >= p['maximum_steps_per_case']:
                        raise RuntimeError('Pilot step budget reached')
                    limits=engine.timestep_limits()
                    dt=select_timestep(limits['elastic_seconds'],limits['viscous_seconds'],p['timestep_cap_s'],safety=case['timestep_safety'])
                    count=min(p['batch_steps'],p['maximum_steps_per_case']-(engine.step-p['start_step']))
                    remaining_log=-np.log1p(-p['axial_engineering_strain'])+np.log1p(-strain[2])
                    axial_rate=min(case['axial_rate_s'],remaining_log/(count*dt))
                    row=engine.diagnostics()
                    rates=loading_rates(row,p,dt,count,axial_rate)
                    for retry in range(p['maximum_timestep_halvings']+1):
                        if dt<p['minimum_timestep_s']:raise RuntimeError('Frozen dt floor reached')
                        engine.spec=replace(engine.spec,timestep=dt)
                        try:
                            engine.advance(count,rate=rates,drag=p['loading_drag_s'])
                            break
                        except RejectedStep as exc:
                            rejected.append(dict(step=engine.step,dt=dt,flags=exc.flags))
                            if exc.flags & ~(8192|4) or retry==p['maximum_timestep_halvings']:raise
                            dt*=.5
                    elapsed+=count*dt
                    logarithm=engine.m.read(engine.buffers['cellLog'],shape=(2,4))
                    delta=logarithm[0,:3].astype(float)+logarithm[1,:3]-origin_log
                    strain=-np.expm1(delta)
                    row=engine.diagnostics()
                    stress=np.array(row['stress']); signed_q=float(stress[2,2]-.5*(stress[0,0]+stress[1,1]))
                    work_residual=(row['cell_work']-initial['cell_work']
                        -(row['friction_dissipation']-initial['friction_dissipation'])
                        -(row['drag_dissipation']-initial['drag_dissipation'])
                        -(row['kinetic_energy']-initial['kinetic_energy'])
                        -(row['elastic_energy_proxy']-initial['elastic_energy_proxy']))
                    row.update(case=case['id'],additional_steps=engine.step-p['start_step'],elapsed_loading_time_s=elapsed,
                        axial_engineering_strain=float(strain[2]),engineering_strain=strain.tolist(),
                        volumetric_engineering_strain=float(strain.sum()),finite_volume_strain=float(-np.expm1(delta.sum())),
                        signed_q_Pa=signed_q,command_rate_s=rates.tolist(),timestep_s=dt,
                        work_proxy_residual_J=work_residual,
                        radial_pressure_error_Pa=[float(stress[i,i]-p['target_pressure_Pa']) for i in range(2)],
                        inertial_number_max=float(case['axial_rate_s']*diameters.max()*np.sqrt(spec.density/max(row['p'],1e-30))),
                        wall_seconds=time.monotonic()-begun)
                    records.append(row);log.write(json.dumps(row)+'\n');log.flush()
                    write_json(target/'latest.json',row);write_json(out/'latest.json',row)
                    if row['additional_steps']%p['snapshot_steps']==0:
                        snapshot(engine,target/'state-latest.h5','triaxial-pilot',elapsed)
                if max(r['inertial_number_max'] for r in records)>p['inertial_number_ceiling']:
                    raise RuntimeError('Frozen inertial-number screen failed')
        except Exception as exc:failure=f'{type(exc).__name__}: {exc}'
        snapshot(engine,target/'state-final.h5','triaxial-pilot',elapsed)
        result={'case':case,'failure':failure,'execution_passed':failure is None,'steps':engine.step-p['start_step'],
            'wall_seconds':time.monotonic()-begun,'initial':initial,'last':records[-1] if records else None,
            'rejections':rejected,'scientific_admission':False,'calibrated_material':False}
        if records:
            result['observed_wall_seconds_per_axial_strain'] = result['wall_seconds']/max(records[-1]['axial_engineering_strain'],1e-30)
        write_json(target/'report.json',result);results.append(result)
        write_json(out/'results-progress.json',{'cases':results})
        del engine
        if failure:break
    freeze=json.loads((out/'execution-freeze.json').read_text())
    write_json(out/'report.json',{'identity':p['identity'],'execution_passed':len(results)==len(p['cases']) and all(r['execution_passed'] for r in results),
        'input_integrity':all(sha256(f)==h for f,h in freeze['inputs'].items()),'cases':results,
        'scope':p['scope'],'scientific_admission':False,'calibrated_material':False})
    return 0 if all(r['execution_passed'] for r in results) else 1


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('protocol',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--worker',action='store_true');a=ap.parse_args();out=a.output.resolve()
    if not a.worker:
        out.mkdir(exist_ok=False,parents=True);p=json.loads(a.protocol.read_text());parent=Path(p['preparation_run'])
        snap=out/'source_snapshot';shutil.copytree(parent/'source_snapshot',snap,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(__file__,snap/Path(__file__).name);shutil.copy2(a.protocol,out/'protocol.json')
        fixed=[a.protocol,parent/'protocol.json',parent/'freeze.json',Path(p['reference_specification'])]
        write_json(out/'queue-freeze.json',{'inputs':{str(Path(f).resolve()):sha256(f) for f in fixed},
            'sources':{str(f.relative_to(snap)):sha256(f) for f in snap.rglob('*') if f.is_file()},
            'future_inputs':'Accepted preparation report, telemetry, final checkpoint and shape hash bound in execution-freeze before GPU execution'})
        with (out/'run.log').open('w') as log:
            r=subprocess.run([sys.executable,str(snap/Path(__file__).name),str(out/'protocol.json'),'--output',str(out),'--worker'],
                env=dict(os.environ,PYTHONPATH=str(snap),SAND_DEM_METAL='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT,check=False)
        write_json(out/'execution.json',{'returncode':r.returncode});return r.returncode
    p=json.loads((out/'protocol.json').read_text());freeze=json.loads((out/'queue-freeze.json').read_text())
    while True:
        if not all(sha256(f)==h for f,h in freeze['inputs'].items()) or not all(sha256(out/'source_snapshot'/f)==h for f,h in freeze['sources'].items()):
            raise RuntimeError('Frozen queue input/source changed')
        parent=Path(p['preparation_run'])
        if any((parent/name).exists() for name in ['controller-transfer.json','user-transfer.json']):
            write_json(out/'queue-status.json',{'status':'stopped-upstream-preparation-transferred','scientific_admission':False})
            return 1
        if (parent/'report.json').exists():
            # Parent writes final report before its execution receipt; wait for both
            # so GPU processes cannot overlap even at the handoff boundary.
            if (parent/'execution.json').exists():break
        write_json(out/'queue-status.json',{'status':'waiting-for-preparation','preparation_run':str(parent),'scientific_admission':False})
        time.sleep(p['poll_seconds'])
    write_json(out/'queue-status.json',{'status':'starting-preparation-audit','scientific_admission':False})
    try:return execute(out,p)
    except Exception as exc:
        write_json(out/'report.json',{'identity':p['identity'],'execution_passed':False,'failure':f'{type(exc).__name__}: {exc}','scientific_admission':False,'calibrated_material':False})
        return 1


if __name__=='__main__':raise SystemExit(main())
