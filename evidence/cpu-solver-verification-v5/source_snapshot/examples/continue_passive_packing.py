"""Frozen passive continuation of a private annealing snapshot on explicit Metal."""
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
from study_annealed_preparation import snapshot


def worker(out):
    p = json.loads((out / 'protocol.json').read_text())
    freeze = json.loads((out / 'freeze.json').read_text())
    assert all(sha256(f) == h for f, h in freeze['inputs'].items())
    assert all(sha256(out / 'source_snapshot' / f) == h for f, h in freeze['sources'].items())
    with h5py.File(p['parent_snapshot']) as f:
        spec = Experiment(**json.loads(f.attrs['experiment']))
        state = {name: f[name][:] for name in f if name not in ('shape_ids', 'scales')}
        state['contacts'] = int(f.attrs['contacts'])
        ids, scales = f['shape_ids'][:], f['scales'][:]
        start = int(f.attrs['step'])
        elapsed = float(f.attrs['elapsed_pseudo_time_s'])
        length, time_unit = float(f.attrs['length_unit']), float(f.attrs['time_unit'])
    library = ShapeLibrary.load(spec.shape_library)
    positions = (state['position'][:, :3].astype(float) + state['positionRemainder'][:, :3]) * length
    cell = state['cell'][0, :3].astype(float) * length
    engine = None
    rows, rejections, restart = [], [], {}
    failure = None
    begun = time.monotonic()
    def observe():
        row = engine.diagnostics()
        row.update(additional_steps=engine.step-start, elapsed_pseudo_time_s=elapsed,
                   wall_seconds=time.monotonic()-begun, phase='passive-construction')
        row['kinetic_proxy_ratio'] = row['kinetic_energy'] / max(row['elastic_energy_proxy'], 1e-30)
        return row
    try:
        engine = Engine(library, spec, positions, cell, ids, scales)
        assert engine.length_unit == length and engine.time_unit == time_unit
        engine.restore(state, start)
        restored = engine.state()
        restart['primary_exact'] = all(np.array_equal(restored[k], state[k]) for k in p['restart_primary_exact'])
        ia = np.lexsort(restored['meta'][:, :3].T[::-1])
        ib = np.lexsort(state['meta'][:, :3].T[::-1])
        keys = np.array_equal(restored['meta'][ia], state['meta'][ib])
        restart['contact_keys_exact'] = keys
        restart['contact_count_exact'] = restored['contacts'] == state['contacts']
        restart['keyed_normals_exact'] = keys and np.array_equal(restored['normalArea'][ia], state['normalArea'][ib])
        restart['keyed_slip_bound'] = keys and np.allclose(restored['slip'][ia], state['slip'][ib], rtol=p['restart_slip_rtol'], atol=p['restart_slip_atol'])
        restart['resultants_bound'] = all(np.allclose(restored[k], state[k], rtol=p['restart_resultant_rtol'], atol=p['restart_resultant_atol']) for k in ['force', 'torque', 'accumulatedRemainder'])
        write_json(out / 'restart-checks.json', restart)
        if not all(restart.values()):
            raise RuntimeError('Frozen restart checks failed')
        rows.append(observe())
        write_json(out / 'initial.json', rows[0])
        with (out / 'telemetry.jsonl').open('w') as log:
            while engine.step-start < p['additional_steps'] and rows[-1]['max_overlap_fraction'] > p['coarse_overlap_stop']:
                if shutil.disk_usage(out).free < p['minimum_free_bytes']:
                    raise RuntimeError('Frozen free-storage reserve reached')
                limits = engine.timestep_limits()
                dt = select_timestep(limits['elastic_seconds'], limits['viscous_seconds'], p['timestep_cap_s'])
                count = min(p['batch_steps'], p['additional_steps']-(engine.step-start))
                for retry in range(p['maximum_timestep_halvings']+1):
                    if dt < p['minimum_timestep_s']:
                        raise RuntimeError('Frozen minimum timestep reached')
                    engine.spec = replace(engine.spec, timestep=dt)
                    try:
                        engine.advance(count, rate=[0., 0., 0.], drag=p['drag_per_step']/dt)
                        break
                    except RejectedStep as exc:
                        rejections.append(dict(step=engine.step, dt=dt, flags=exc.flags))
                        if exc.flags & ~(8192 | 4) or retry == p['maximum_timestep_halvings']:
                            raise
                        dt *= .5
                elapsed += count*dt
                row = observe()
                row['timestep_s'] = dt
                rows.append(row)
                log.write(json.dumps(row)+'\n')
                log.flush()
                write_json(out / 'latest.json', row)
                if row['additional_steps'] % p['snapshot_steps'] == 0:
                    snapshot(engine, out / 'state-latest.h5', 'passive-construction', elapsed)
                    print('passive', row['additional_steps'], 'overlap', row['max_overlap_fraction'], 'imbalance', row['unbalanced_force'], 'kinetic/proxy', row['kinetic_proxy_ratio'], flush=True)
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        print(failure, flush=True)
    finally:
        if engine is not None:
            snapshot(engine, out / 'state-final.h5', 'passive-construction', elapsed)
        trend = None
        if rows and rows[-1]['additional_steps'] >= p['trend_window_steps']:
            tail = [r for r in rows if r['additional_steps'] >= rows[-1]['additional_steps']-p['trend_window_steps']]
            first, last = tail[0], tail[-1]
            decrease = 1-last['max_overlap_fraction']/max(first['max_overlap_fraction'], 1e-30)
            trend = {'window_steps':last['additional_steps']-first['additional_steps'],
                     'initial_overlap':first['max_overlap_fraction'], 'final_overlap':last['max_overlap_fraction'],
                     'relative_overlap_decrease':decrease,
                     'maximum_force_imbalance':max(r['unbalanced_force'] for r in tail),
                     'maximum_kinetic_proxy_ratio':max(r['kinetic_proxy_ratio'] for r in tail)}
            trend['stationary_trap_screen'] = (abs(decrease) <= p['plateau_relative_change'] and
                trend['maximum_force_imbalance'] <= p['force_tolerance'] and
                trend['maximum_kinetic_proxy_ratio'] <= p['kinetic_tolerance'] and
                last['max_overlap_fraction'] > p['coarse_overlap_stop'])
        checks = {'restart':bool(restart) and all(restart.values()), 'execution':failure is None,
                  'inputs':all(sha256(f)==h for f,h in freeze['inputs'].items()),
                  'sources':all(sha256(out/'source_snapshot'/f)==h for f,h in freeze['sources'].items())}
        cleared = bool(rows) and rows[-1]['max_overlap_fraction'] <= p['coarse_overlap_stop']
        report = {'identity':p['identity'], 'software_checks':checks, 'software_passed':all(checks.values()),
                  'coarse_overlap_below_one_percent':cleared, 'scientific_admission':False,
                  'physical_handoff':False, 'failure':failure, 'initial':rows[0] if rows else None,
                  'last':rows[-1] if rows else None, 'trend':trend, 'rejections':rejections,
                  'wall_seconds':time.monotonic()-begun, 'scope':p['scope']}
        write_json(out / 'report.json', report)
    return 0 if report['software_passed'] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('protocol', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--worker', action='store_true')
    args = ap.parse_args()
    out = args.output.resolve()
    if args.worker:
        return worker(out)
    out.mkdir(exist_ok=False, parents=True)
    root = Path(__file__).resolve().parents[1]
    p = json.loads(args.protocol.read_text())
    parent = Path(p['parent_snapshot']).parent
    # Use the actual solver bytes from the preceding trial, not a revised solver.
    snap = out / 'source_snapshot'
    shutil.copytree(parent / 'source_snapshot' / 'sand_atlas_dem', snap / 'sand_atlas_dem', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, snap / Path(__file__).name)
    shutil.copy2(parent / 'source_snapshot' / 'study_annealed_preparation.py', snap / 'study_annealed_preparation.py')
    shutil.copy2(args.protocol, out / 'protocol.json')
    with h5py.File(p['parent_snapshot']) as f:
        shape = json.loads(f.attrs['experiment'])['shape_library']
        assert sha256(shape) == f.attrs['shape_sha256']
    inputs = [args.protocol, p['parent_snapshot'], shape, parent / 'report.json', parent / 'prospective-freeze.json']
    write_json(out / 'freeze.json', {'inputs':{str(Path(f).resolve()):sha256(f) for f in inputs},
        'sources':{str(f.relative_to(snap)):sha256(f) for f in snap.rglob('*') if f.is_file()}})
    with (out / 'run.log').open('w') as log:
        run = subprocess.run([sys.executable, str(snap / Path(__file__).name), str(out / 'protocol.json'), '--output', str(out), '--worker'], cwd=root,
            env=dict(os.environ, PYTHONPATH=str(snap), SAND_DEM_METAL='1', PYTHONUNBUFFERED='1'), stdout=log, stderr=subprocess.STDOUT, check=False)
    write_json(out / 'execution.json', {'returncode':run.returncode})
    return run.returncode


if __name__ == '__main__':
    raise SystemExit(main())
