"""Minimal static GPU reproducer for non-repulsive deep-overlap surface law."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import trimesh

from sand_atlas_dem.geometry import ShapeLibrary, from_mesh
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import sha256, write_json


def continuum(x):
    """Return E/(pi kn R^4), F_separating/(pi kn R^3); equal spheres."""
    if x <= 0 or x >= 2:
        return 0., 0.
    if x <= 1:
        return x*x/3-x**3/4, .75*x*x-2*x/3
    u = 2-x
    integral = u**3/3-u**4/4
    return integral/x, (x*(u*u-u**3)+integral)/(x*x)


def execute(out):
    p = json.loads((out/'protocol.json').read_text())
    freeze = json.loads((out/'freeze.json').read_text())
    assert sha256(out/'protocol.json') == freeze['protocol_sha256']
    radius, kn = p['radius_m'], p['normal_stiffness_Pa_per_m']
    mesh = trimesh.creation.icosphere(subdivisions=p['icosphere_subdivisions'], radius=radius)
    shape = from_mesh(mesh, unit='m', samples=p['samples'], seed=p['seed'], resolution=p['sdf_resolution'])
    library = ShapeLibrary([shape])
    # Synthetic, reproducible fixture only, not downloaded/private source geometry.
    library.save(out/'fixture.h5')
    force_unit, energy_unit = np.pi*kn*radius**3, np.pi*kn*radius**4
    spec = Experiment(shape_library=str(out/'fixture.h5'), particles=2, contact_geometry='mesh',
                      timestep=1e-10, timestep_policy='active-contact', normal_stiffness=kn,
                      tangential_stiffness=.5*kn, friction=0., normal_viscosity=0.,
                      max_overlap_fraction=.5, contacts_per_particle=p['samples'])
    results = []
    metal = None
    for x in p['separation_over_radius']:
        positions = np.array([[.01-x*radius/2,.01,.01],[.01+x*radius/2,.01,.01]])
        engine = Engine(library, spec, positions, [.02]*3, [0,0], [1.,1.], metal=metal)
        engine.initialize()
        metal = engine.m
        state = engine.state()
        force = state['force'][1,:3]*engine.force_unit
        diagnostics = engine.diagnostics()
        energy, radial = continuum(x)
        measured_energy = diagnostics['elastic_energy_proxy']
        results.append({'d_over_R':x,'force_on_right_N':force.tolist(),
                        'expected_sphere_separating_force_N':radial*force_unit,
                        'energy_J':measured_energy,'expected_sphere_energy_J':energy*energy_unit,
                        'force_error_normalized':float(abs(force[0]/force_unit-radial)),
                        'energy_error_normalized':float(abs(measured_energy/energy_unit-energy)),
                        'radially_repulsive':bool(force[0]>0),'diagnostics':diagnostics})
        print('d/R',x,'radial force',force[0],'analytic',radial*force_unit,flush=True)
        del engine
    checks = {'force_analytic_agreement':all(r['force_error_normalized']<=p['force_error_over_pi_kn_R3'] for r in results),
              'energy_analytic_agreement':all(r['energy_error_normalized']<=p['energy_error_over_pi_kn_R4'] for r in results),
              'predicted_signs':all((r['force_on_right_N'][0]>0)==(r['d_over_R']>8/9) for r in results),
              'source_integrity':all(sha256(out/'source_snapshot'/f)==h for f,h in freeze['sources'].items()),
              'protocol_integrity':sha256(out/'protocol.json')==freeze['protocol_sha256']}
    write_json(out/'report.json', {'identity':p['identity'],'checks':checks,'reproducer_passed':all(checks.values()),
                                  'repulsion_requirement_met':all(r['radially_repulsive'] for r in results),
                                  'fixture_sha256':sha256(out/'fixture.h5'),'results':results,'scope':p['scope']})
    return 0 if all(checks.values()) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('protocol',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--worker',action='store_true')
    a = ap.parse_args()
    out = a.output.resolve()
    if a.worker:
        return execute(out)
    root = Path(__file__).resolve().parents[1]
    out.mkdir(parents=True,exist_ok=False)
    snap = out/'source_snapshot'
    shutil.copytree(root/'src/sand_atlas_dem',snap/'sand_atlas_dem',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,snap/Path(__file__).name)
    shutil.copy2(a.protocol,out/'protocol.json')
    write_json(out/'freeze.json', {'protocol_sha256':sha256(out/'protocol.json'),
                                  'sources':{str(f.relative_to(snap)):sha256(f) for f in snap.rglob('*') if f.is_file()}})
    with (out/'run.log').open('w') as log:
        r = subprocess.run([sys.executable,str(snap/Path(__file__).name),str(out/'protocol.json'),'--output',str(out),'--worker'],
                           cwd=root,env=dict(os.environ,PYTHONPATH=str(snap),SAND_DEM_METAL='1',PYTHONUNBUFFERED='1'),
                           stdout=log,stderr=subprocess.STDOUT,check=False)
    write_json(out/'execution.json',{'returncode':r.returncode})
    return r.returncode


if __name__ == '__main__':
    raise SystemExit(main())
