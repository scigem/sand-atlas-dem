"""Interactively inspect an annealing snapshot using the existing PyVista renderer."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import sha256, write_json
from sand_atlas_dem.viewer import FIELDS, VECTORS, Renderer, ViewOptions, particle_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--color-by', choices=FIELDS, default='speed')
    parser.add_argument('--vectors', choices=VECTORS, default='none')
    parser.add_argument('--screenshot-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    with h5py.File(args.snapshot) as f:
        if f.attrs['format'] != 'sand-dem-annealing-state/1':
            raise ValueError('Expected an annealing-state snapshot')
        spec = Experiment(**json.loads(f.attrs['experiment']))
        if sha256(spec.shape_library) != f.attrs['shape_sha256']:
            raise ValueError('Shape library identity mismatch')
        state = {k:f[k][:] for k in f if k not in ('shape_ids', 'scales')}
        state['contacts'] = int(f.attrs['contacts'])
        length, time = float(f.attrs['length_unit']), float(f.attrs['time_unit'])
        engine = SimpleNamespace(ids=f['shape_ids'][:], scales=f['scales'][:], length_unit=length,
            time_unit=time, mass_unit=spec.density*length**3, force_unit=spec.normal_stiffness*length**3,
            step=int(f.attrs['step']), spec=spec, state=lambda:state)
        elapsed = float(f.attrs['elapsed_pseudo_time_s'])
    library = ShapeLibrary.load(spec.shape_library)
    frame = particle_frame(engine)
    frame['time'] = elapsed  # Adaptive construction timesteps: never step*last dt.
    renderer = Renderer(SimpleNamespace(engine=engine, library=library), args.output,
                        ViewOptions(field=args.color_by, vectors=args.vectors))
    pv = renderer.pv
    plotter = pv.Plotter(off_screen=args.screenshot_only, window_size=(1200, 900),
                         title='Sand Atlas DEM — saved packing, step '+str(engine.step))
    renderer.plotters.append(plotter)
    renderer.initial_cell = frame['cell'].copy()
    plotter.set_background('#18212c')
    plotter.add_axes()
    renderer._draw(frame)
    renderer._camera(plotter, (1, 1, 1))
    plotter.add_text('Saved construction state | not physical equilibrium', position='lower_left',
                     font_size=10, color='white')
    write_json(args.output/'snapshot-view.json', {'snapshot':str(args.snapshot.resolve()),
        'snapshot_sha256':sha256(args.snapshot), 'shape_sha256':sha256(spec.shape_library),
        'step':engine.step, 'elapsed_pseudo_time_s':elapsed, 'field':args.color_by,
        'geometry':'Original grain triangles at saved physical poses; no ghost copies',
        'scientific_admission':False})
    plotter.show(auto_close=args.screenshot_only, screenshot=str(args.output/'packing.png'))
    renderer.frames = 1
    renderer.last_step = engine.step
    renderer.close()


if __name__ == '__main__':
    main()
