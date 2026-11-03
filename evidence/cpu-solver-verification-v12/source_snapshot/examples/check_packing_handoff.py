"""Frozen full-quadrature physical handoff screen and bounded equilibration."""
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
from study_annealed_preparation import make_engine, snapshot
from packing_pressure_control import pressure_rates


def worker(out):
    p = json.loads((out/'protocol.json').read_text())
    freeze = json.loads((out/'freeze.json').read_text())
    assert all(sha256(f)==h for f,h in freeze['inputs'].items())
    with h5py.File(p['original_checkpoint']) as f:
        original = Experiment(**json.loads(f.attrs['experiment']))
    with h5py.File(p['parent_snapshot']) as f:
        length = float(f.attrs['length_unit'])
        x = (f['position'][:, :3].astype(float)+f['positionRemainder'][:, :3])*length
        cell = f['cell'][0, :3].astype(float)*length
        q = f['rotation'][:].astype(float)+f['rotationRemainder'][:]
        q /= np.linalg.norm(q, axis=1)[:, None]
        ids, scales = f['shape_ids'][:], f['scales'][:]
        parent_spec = Experiment(**json.loads(f.attrs['experiment']))
        parent_step = int(f.attrs['step'])
        parent_time = float(f.attrs['elapsed_pseudo_time_s'])
        parent_time_unit = float(f.attrs['time_unit'])
        parent_state = {k:f[k][:] for k in f if k not in ('shape_ids', 'scales')}
        parent_state['contacts'] = int(f.attrs['contacts'])
    library = ShapeLibrary.load(original.shape_library)
    overlap_guard = p.get('overlap_guard', original.max_overlap_fraction)
    equilibrium_overlap = p.get('equilibrium_overlap_limit', original.max_overlap_fraction)
    spec = replace(original, timestep_policy='active-contact', timestep=p['minimum_timestep_s'],
                   max_overlap_fraction=overlap_guard)
    engine = None
    failure = None
    rejections, rows = [], []
    converged, elapsed = 0, 0.
    started = time.monotonic()
    start_step = 0
    restart = {}
    step_limit = p['equilibration_steps']
    if step_limit is not None and (type(step_limit) is not int or step_limit <= 0):
        raise ValueError('equilibration_steps must be a positive integer or null for no cap')
    try:
        if not 0 <= p['taper_overlap'] < p['hold_overlap'] < overlap_guard:
            raise ValueError('Require taper < compression hold < overlap guard')
        if shutil.disk_usage(out).free < p['minimum_free_bytes']:
            raise RuntimeError('Frozen storage reserve reached')
        if p.get('resume_physical_state', False):
            start_step, elapsed = parent_step, parent_time
            for key in ['shape_library', 'contact_geometry', 'contact_normal', 'density',
                        'normal_stiffness', 'tangential_stiffness', 'friction', 'normal_viscosity',
                        'periodic', 'planes', 'boundary_box', 'timestep_policy', 'max_overlap_fraction']:
                if getattr(parent_spec, key) != getattr(spec, key):
                    raise ValueError('Physical restart setting mismatch: '+key)
            engine = Engine(library, parent_spec, x, cell, ids, scales)
            if engine.length_unit != length or engine.time_unit != parent_time_unit:
                raise ValueError('Physical restart unit mismatch')
            engine.restore(parent_state, parent_step)
            # A dt=0 Coulomb rebuild may reproject a rounded spring and report a
            # tiny return loss. Restart must not add that to cumulative work.
            replay = {}
            rebuilt = engine.state()
            for name in ['accumulated', 'accumulatedRemainder']:
                replay[name] = float(np.max(abs(rebuilt[name].astype(float)-parent_state[name])))
                engine.buffers[name].copy_from_numpy(parent_state[name])
            write_json(out/'restart-bookkeeping.json', {
                'dt_zero_rebuild_max_abs_normalized':replay,
                'action':'Preserve checkpoint cumulative work/loss high and residual arrays exactly; no dynamic state or friction-history reset'})
            restored = engine.state()
            restart['primary_exact'] = all(np.array_equal(restored[k], parent_state[k]) for k in p['restart_primary_exact'])
            ia = np.lexsort(restored['meta'][:, :3].T[::-1])
            ib = np.lexsort(parent_state['meta'][:, :3].T[::-1])
            keys = np.array_equal(restored['meta'][ia], parent_state['meta'][ib])
            restart['contact_keys_exact'] = keys
            restart['contact_count_exact'] = restored['contacts'] == parent_state['contacts']
            restart['keyed_normals_exact'] = keys and np.array_equal(restored['normalArea'][ia], parent_state['normalArea'][ib])
            restart['keyed_slip_bound'] = keys and np.allclose(restored['slip'][ia], parent_state['slip'][ib], rtol=p['restart_slip_rtol'], atol=p['restart_slip_atol'])
            restart['resultants_bound'] = all(np.allclose(restored[k], parent_state[k], rtol=p['restart_resultant_rtol'], atol=p['restart_resultant_atol']) for k in ['force', 'torque', 'accumulatedRemainder'])
            write_json(out/'restart-checks.json', restart)
            if not all(restart.values()):
                raise RuntimeError('Frozen physical restart checks failed')
        else:
            engine = make_engine(library, spec, x, cell, ids, scales, q)
        initial = engine.diagnostics()
        write_json(out/'handoff.json', {'diagnostics':initial,
            'full_quadrature':True, 'physical_properties_restored':True,
            'overlap_guard':overlap_guard, 'original_overlap_limit':original.max_overlap_fraction,
            'kinematics':'Exact physical restart; kinematics and keyed friction history retained'
                if p.get('resume_physical_state', False) else 'Retain poses; explicitly reset artificial velocities, angular momentum and friction history',
            'capacity_retries':getattr(engine, 'capacity_retries', []),
            'controller':{k:p[k] for k in ['taper_overlap', 'hold_overlap', 'overlap_guard'] if k in p}})
        print('Full-quadrature initial overlap', initial['max_overlap_fraction'],
              'guard', overlap_guard, 'pressure', initial['p'], flush=True)
        snapshot(engine, out/'physical-initial.h5', 'physical-equilibration', elapsed)
        with (out/'telemetry.jsonl').open('w') as log:
            while step_limit is None or engine.step-start_step < step_limit:
                if shutil.disk_usage(out).free < p['minimum_free_bytes']:
                    raise RuntimeError('Frozen storage reserve reached')
                row = engine.diagnostics()
                limits = engine.timestep_limits()
                dt = select_timestep(limits['elastic_seconds'], limits['viscous_seconds'], p['timestep_cap_s'])
                count = p['batch_steps'] if step_limit is None else min(p['batch_steps'], step_limit-(engine.step-start_step))
                command, taper = pressure_rates(row['stress'], row['max_overlap_fraction'], p, count*dt)
                for retry in range(p['maximum_timestep_halvings']+1):
                    if dt < p['minimum_timestep_s']:
                        raise RuntimeError('Frozen minimum timestep reached')
                    engine.spec = replace(engine.spec, timestep=dt)
                    try:
                        engine.advance(count, rate=command, drag=p['drag_per_step']/dt)
                        break
                    except RejectedStep as exc:
                        rejections.append(dict(step=engine.step, dt=dt, flags=exc.flags))
                        if exc.flags & ~(8192|4) or retry == p['maximum_timestep_halvings']:
                            raise
                        dt *= .5
                elapsed += count*dt
                row = engine.diagnostics()
                row.update(timestep_s=dt, elapsed_pseudo_time_s=elapsed, wall_seconds=time.monotonic()-started,
                           additional_steps=engine.step-start_step, command_rate_s=command.tolist(),
                           pressure_control=p.get('pressure_control', 'mean'), compression_taper=float(taper),
                           normal_stress_error_Pa=(np.diag(row['stress'])-p['target_pressure_Pa']).tolist())
                rows.append(row)
                log.write(json.dumps(row)+'\n'); log.flush()
                write_json(out/'latest.json', row)
                if engine.step % p['snapshot_steps'] == 0:
                    snapshot(engine, out/'state-latest.h5', 'physical-equilibration', elapsed)
                    print('physical', engine.step, 'overlap', row['max_overlap_fraction'], 'pressure', row['p'], flush=True)
                ok = (np.max(abs(np.diag(row['stress'])/p['target_pressure_Pa']-1)) <= p['pressure_tolerance']
                      and row['kinetic_energy']/max(row['elastic_energy_proxy'], 1e-30) <= p['kinetic_tolerance']
                      and row['unbalanced_force'] <= p['force_tolerance']
                      and row['max_overlap_fraction'] <= equilibrium_overlap)
                converged = converged+1 if ok else 0
                if converged >= p['converged_samples']:
                    break
            if converged < p['converged_samples']:
                raise RuntimeError('Physical equilibrium not reached within frozen budget')
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        print(failure, flush=True)
    finally:
        if engine is not None:
            snapshot(engine, out/'state-final.h5', 'physical-equilibration', elapsed)
        write_json(out/'report.json', {'identity':p['identity'], 'passed':failure is None,
            'equilibration_step_limit':step_limit,
            'overlap_guard':overlap_guard, 'equilibrium_overlap_limit':equilibrium_overlap,
            'original_overlap_limit':original.max_overlap_fraction,
            'original_overlap_limit_met':bool(rows) and rows[-1]['max_overlap_fraction'] <= original.max_overlap_fraction,
            'failure':failure, 'full_quadrature_handoff':engine is not None,
            'physical_steps':engine.step-start_step if engine else 0,
            'total_physical_step':engine.step if engine else 0, 'restart_checks':restart,
            'consecutive_equilibrium_samples':converged,
            'last':rows[-1] if rows else None, 'rejections':rejections, 'wall_seconds':time.monotonic()-started,
            'source_integrity':all(sha256(out/'source_snapshot'/f)==h for f,h in freeze['sources'].items()),
            'input_integrity':all(sha256(f)==h for f,h in freeze['inputs'].items()),
            'scientific_admission':False, 'scope':p['scope']})
    return 0 if failure is None else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('protocol', type=Path); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--worker', action='store_true'); args = ap.parse_args()
    out = args.output.resolve()
    if args.worker:
        return worker(out)
    out.mkdir(parents=True, exist_ok=False)
    p = json.loads(args.protocol.read_text())
    parent = Path(p['parent_snapshot']).parent
    snap = out/'source_snapshot'
    shutil.copytree(parent/'source_snapshot', snap, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, snap/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('packing_pressure_control.py'), snap/'packing_pressure_control.py')
    shutil.copy2(args.protocol, out/'protocol.json')
    with h5py.File(p['original_checkpoint']) as f:
        shape = json.loads(f.attrs['experiment'])['shape_library']
        assert sha256(shape) == f.attrs['shape_sha256']
    with h5py.File(p['parent_snapshot']) as f:
        parent_shape = json.loads(f.attrs['experiment'])['shape_library']
        assert sha256(parent_shape) == f.attrs['shape_sha256']
    inputs = [args.protocol, p['parent_snapshot'], p['original_checkpoint'], shape]
    write_json(out/'freeze.json', {'inputs':{str(Path(f).resolve()):sha256(f) for f in inputs},
        'sources':{str(f.relative_to(snap)):sha256(f) for f in snap.rglob('*') if f.is_file()}})
    with (out/'run.log').open('w') as log:
        run = subprocess.run([sys.executable, str(snap/Path(__file__).name), str(out/'protocol.json'), '--output', str(out), '--worker'],
            cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, PYTHONPATH=str(snap), SAND_DEM_METAL='1', PYTHONUNBUFFERED='1'), stdout=log, stderr=subprocess.STDOUT, check=False)
    write_json(out/'execution.json', {'returncode':run.returncode})
    return run.returncode


if __name__ == '__main__':
    raise SystemExit(main())
