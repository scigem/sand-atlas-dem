"""Serial Metal prototype checks on an original concave mesh, beyond spheres."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.gpu import Engine
from sand_atlas_dem.spec import Experiment
from sand_atlas_dem.util import quat_matrix

module_spec=importlib.util.spec_from_file_location('source_normal_study',Path(__file__).resolve().parents[1]/'examples/study_source_normals.py')
study=importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(study)
pytestmark=[pytest.mark.metal,pytest.mark.skipif(os.environ.get('SAND_DEM_METAL')!='1',reason='Explicit serial Metal execution required')]


def test_concave_source_normals_independent_forces_torques_and_objectivity():
    library=ShapeLibrary.load('assets/generated/concave-prism.h5')
    spec=Experiment(particles=2,contact_geometry='mesh',contact_normal='source-surface-experimental',
                    friction=0.,timestep=1e-10,timestep_policy='active-contact',max_overlap_fraction=.5,
                    contacts_per_particle=1024)
    cell=np.full(3,.02);centre=cell*.5
    x=np.array([centre,centre+[.0007,.0001,.0001]])
    q=np.array([[0.,0.,0.,1.],[0.,0.,np.sin(.15),np.cos(.15)]])
    ids=[0,0];scales=[1.,1.];metal=None
    forces=[];torques=[]
    turn=np.array([.2,.1,-.3,.9]);turn/=np.linalg.norm(turn);Q=quat_matrix(turn)
    for rotated in [False,True]:
        xx=(x-centre) @ Q.T+centre if rotated else x
        qq=np.array([study.qmul(turn,qi) for qi in q]) if rotated else q
        engine=Engine(library,spec,xx,cell,ids,scales,metal=metal)
        engine.buffers['rotation'].copy_from_numpy(qq.astype(np.float32));engine.initialize();metal=engine.m
        state=engine.state();f=state['force'][:,:3]*engine.force_unit;t=state['torque'][:,:3]*engine.energy_unit
        rf,rt,_=study.reference(library,spec,xx,qq,ids,scales,engine.length_unit)
        assert engine.diagnostics()['active_surface_contacts']>0
        assert np.allclose(f,rf,rtol=3e-4,atol=1e-10)
        assert np.allclose(t,rt,rtol=3e-4,atol=1e-14)
        assert np.linalg.norm(f.sum(0))<1e-5*np.linalg.norm(f)
        origin_torque=t.sum(0)+np.cross(xx-xx[0],f).sum(0)
        assert np.linalg.norm(origin_torque)<1e-4*(np.linalg.norm(t)+.001*np.linalg.norm(f))
        forces.append(f);torques.append(t)
        del engine
    assert np.allclose(forces[1],forces[0] @ Q.T,rtol=.001,atol=1e-9)
    assert np.allclose(torques[1],torques[0] @ Q.T,rtol=.001,atol=1e-12)
