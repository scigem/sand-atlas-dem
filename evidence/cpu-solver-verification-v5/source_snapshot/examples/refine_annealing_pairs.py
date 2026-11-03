"""Bounded quadrature/perturbation diagnostic; actual Metal dynamics only."""
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

from sand_atlas_dem.annealing import construction_quadrature, select_timestep
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.gpu import Engine, RejectedStep
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import sha256, write_json


def pose(engine):
    s = engine.state()
    x = (s['position'][:, :3].astype(float) + s['positionRemainder'][:, :3]) * engine.length_unit
    q = s['rotation'].astype(float) + s['rotationRemainder']
    q /= np.linalg.norm(q, axis=1)[:, None]
    return x, q


def rotate_x(q, angle):
    # Left multiplication: a spatial x-axis increment, xyzw convention.
    x, y, z, w = q
    s, c = np.sin(angle/2), np.cos(angle/2)
    return np.array([c*x+s*w, c*y-s*z, c*z+s*y, c*w-s*x])


def execute(out):
    p = json.loads((out/'protocol.json').read_text())
    freeze = json.loads((out/'freeze.json').read_text())
    assert all(sha256(f) == h for f, h in freeze['inputs'].items())
    with h5py.File(p['state']) as f:
        spec = Experiment(**json.loads(f.attrs['experiment']))
        length = float(f.attrs['length_unit'])
        ids, scales = f['shape_ids'][:], f['scales'][:]
        positions = (f['position'][:, :3].astype(float)+f['positionRemainder'][:, :3])*length
        rotations = f['rotation'][:].astype(float)+f['rotationRemainder'][:]
        rotations /= np.linalg.norm(rotations, axis=1)[:, None]
        cell = f['cell'][0, :3].astype(float)*length
    coarse = ShapeLibrary.load(spec.shape_library)
    full = ShapeLibrary.load(p['full_library'])
    assert [s.id for s in coarse.shapes] == [s.id for s in full.shapes]
    libraries = {1024: coarse}
    for count in p['levels'][1:]:
        shaped = [construction_quadrature(s, count, p['seed']+i) for i, s in enumerate(full.shapes)]
        libraries[count] = ShapeLibrary([s for s, _ in shaped])
        np.savez_compressed(out/f'node-map-{count}.npz', **{str(i): ix for i, (_, ix) in enumerate(shaped)})
    del full
    results = []
    metal = None

    def relax(label, bodies, count, x, q, steps):
        nonlocal metal
        if shutil.disk_usage(out).free < p['minimum_free_bytes']:
            raise RuntimeError('Frozen free-disk floor reached')
        engine = Engine(libraries[count], replace(spec, particles=2, contacts_per_particle=max(2048, count)),
                        x, cell, ids[bodies], scales[bodies], metal=metal)
        engine.buffers['rotation'].copy_from_numpy(q.astype(np.float32))
        engine.initialize()
        metal = engine.m
        rows = [engine.diagnostics()]
        np.savez_compressed(out/f'{label}-initial-pose.npz', positions_m=x, rotations_xyzw=q,
                            cell_m=cell, shape_ids=ids[bodies], scales=scales[bodies])
        elapsed = 0.
        rejections = []
        failure = None
        try:
            while engine.step < steps:
                limits = engine.timestep_limits()
                dt = select_timestep(limits['elastic_seconds'], limits['viscous_seconds'], p['timestep_cap_s'])
                amount = min(p['batch_steps'], steps-engine.step)
                for retry in range(9):
                    if dt < p['minimum_timestep_s']:
                        raise RuntimeError('Frozen timestep floor reached')
                    engine.spec = replace(engine.spec, timestep=dt)
                    try:
                        engine.advance(amount, drag=p['drag_per_step']/dt)
                        break
                    except RejectedStep as exc:
                        rejections.append({'step': engine.step, 'dt': dt, 'flags': exc.flags})
                        if exc.flags & ~(8192 | 4) or retry == 8:
                            raise
                        dt *= .5
                elapsed += amount*dt
                if engine.step % p['diagnostic_steps'] == 0:
                    row = engine.diagnostics()
                    row['explicit_elapsed_time_s'] = elapsed
                    rows.append(row)
        except Exception as exc:
            failure = f'{type(exc).__name__}: {exc}'
        x, q = pose(engine)
        np.savez_compressed(out/f'{label}-final-pose.npz', positions_m=x, rotations_xyzw=q,
                            cell_m=cell, shape_ids=ids[bodies], scales=scales[bodies])
        # Preserve full residuals/history as private restart evidence; no geometry copies.
        state = engine.state()
        np.savez_compressed(out/f'{label}-final-state.npz', **state)
        final = engine.diagnostics()
        result = {'label': label, 'bodies': bodies, 'samples': count, 'steps': engine.step,
                  'failure': failure, 'rejections': rejections, 'initial': rows[0], 'final': final,
                  'history': rows, 'explicit_elapsed_time_s': elapsed,
                  'length_unit_m': engine.length_unit}
        results.append(result)
        write_json(out/'latest.json', {'results': results})
        print(label, 'overlap', rows[0]['max_overlap_fraction'], '->', final['max_overlap_fraction'],
              'imbalance', final['unbalanced_force'], 'failure', failure, flush=True)
        del engine
        return x, q

    for bodies in p['pairs']:
        a, b = bodies
        branch = positions[b]-positions[a]
        branch -= np.round(branch/cell)*cell
        x = np.array([cell*.5-branch*.5, cell*.5+branch*.5])
        prefix = f'{a}-{b}'
        x0, q0 = relax(f'{prefix}-1024-control', bodies, 1024, x, rotations[bodies], p['steps'])
        finest = None
        for count in p['levels'][1:]:
            finest = relax(f'{prefix}-{count}-refined', bodies, count, x0, q0, p['steps'])
        x, q = finest
        diameter = np.mean([(6*coarse.shapes[int(ids[i])].volume/np.pi)**(1/3)*scales[i] for i in bodies])
        radial = x[1]-x[0]
        radial /= np.linalg.norm(radial)
        for kind in ['translation', 'rotation']:
            for sign in [-1, 1]:
                xp, qp = x.copy(), q.copy()
                if kind == 'translation':
                    xp[1] += sign*p['translation_over_diameter']*diameter*radial
                else:
                    qp[1] = rotate_x(qp[1], sign*p['rotation_radians'])
                relax(f'{prefix}-{kind}-{sign:+d}', bodies, p['levels'][-1], xp, qp, p['perturbation_steps'])
    checks = {'execution': all(r['failure'] is None for r in results),
              'input_integrity': all(sha256(f) == h for f, h in freeze['inputs'].items()),
              'source_integrity': all(sha256(out/'source_snapshot'/f) == h for f, h in freeze['sources'].items())}
    write_json(out/'report.json', {'identity': p['identity'], 'checks': checks, 'passed': all(checks.values()),
                                  'results': results, 'scope': p['scope']})
    return 0 if all(checks.values()) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('protocol', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--worker', action='store_true')
    args = ap.parse_args()
    out = args.output.resolve()
    if args.worker:
        return execute(out)
    root = Path(__file__).resolve().parents[1]
    out.mkdir(exist_ok=False, parents=True)
    snap = out/'source_snapshot'
    shutil.copytree(root/'src/sand_atlas_dem', snap/'sand_atlas_dem', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, snap/Path(__file__).name)
    shutil.copy2(args.protocol, out/'protocol.json')
    p = json.loads(args.protocol.read_text())
    with h5py.File(p['state']) as f:
        shape = json.loads(f.attrs['experiment'])['shape_library']
        assert sha256(shape) == f.attrs['shape_sha256']
    inputs = [args.protocol, p['state'], p['full_library'], p['prior_report'], shape]
    write_json(out/'freeze.json', {'inputs': {str(Path(f).resolve()): sha256(f) for f in inputs},
                                  'sources': {str(f.relative_to(snap)): sha256(f) for f in snap.rglob('*') if f.is_file()}})
    with (out/'run.log').open('w') as log:
        r = subprocess.run([sys.executable, str(snap/Path(__file__).name), str(out/'protocol.json'), '--output', str(out), '--worker'],
                           cwd=root, env=dict(os.environ, PYTHONPATH=str(snap), SAND_DEM_METAL='1', PYTHONUNBUFFERED='1'),
                           stdout=log, stderr=subprocess.STDOUT, check=False)
    write_json(out/'execution.json', {'returncode': r.returncode})
    return r.returncode


if __name__ == '__main__':
    raise SystemExit(main())
