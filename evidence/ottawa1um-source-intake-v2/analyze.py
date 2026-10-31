"""CPU-only source topology/unit crosswalk diagnostics; no preparation or admission."""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import trimesh

from sand_atlas_dem.util import read_json, sha256, write_json

out=Path(__file__).resolve().parent
source_root=out.parent/'ottawa1um-source-intake-v1'
archive=source_root/'private/meshes_30.zip'
table=source_root/'source_snapshot/sand-atlas/assets/sands/ottawa-0000008.csv'
metadata_path=source_root/'source_snapshot/sand-atlas/_data/json/ottawa-0000008.json'
receipt=read_json(source_root/'download-receipt.json')
inputs={str(p):sha256(p) for p in (archive,table,metadata_path,source_root/'download-receipt.json',source_root/'local-source-receipt.json',Path(__file__))}
write_json(out/'prospective-freeze.json',dict(input_sha256=inputs,scope=__doc__,acceptance='No numerical acceptance target or material admission. Preserve every member, its signed-volume comparisons, topology diagnostics and unmatched inventories. Exact coordinate welding only; no repairs or source execution.',numpy=np.__version__,trimesh=trimesh.__version__))
assert inputs[str(archive)]==receipt['sha256']
with table.open(newline='') as f:
    descriptors=list(csv.DictReader(f))
volumes=np.array([float(d['Volume (µm³)']) for d in descriptors])
assert np.all(np.isfinite(volumes)) and np.all(volumes>0)
rows=[]
with zipfile.ZipFile(archive) as z:
    members=z.infolist()
    assert len({m.filename for m in members})==len(members)
    assert sum(m.file_size for m in members)<=512*1024**2
    for member in members:
        assert not member.is_dir() and Path(member.filename).name==member.filename and member.filename.endswith('.stl')
        assert member.file_size<=10*1024**2
        blob=z.read(member)
        mesh=trimesh.load_mesh(io.BytesIO(blob),file_type='stl',process=False)
        vertices,inverse=np.unique(mesh.vertices,axis=0,return_inverse=True)
        mesh=trimesh.Trimesh(vertices,inverse[mesh.faces],process=False)
        raw_volume=float(mesh.volume)
        differences=np.abs(raw_volume/volumes-1)
        order=np.argsort(differences)
        components=len(mesh.split(only_watertight=False))
        topology=bool(components==1 and mesh.is_watertight and mesh.is_winding_consistent and raw_volume>0 and np.all(mesh.area_faces>0))
        row=dict(member=member.filename,sha256=hashlib.sha256(blob).hexdigest(),bytes=member.file_size,vertices=len(vertices),faces=len(mesh.faces),signed_raw_volume=raw_volume,raw_bounds=None if mesh.bounds is None else mesh.bounds.tolist(),components=components,watertight=bool(mesh.is_watertight),winding_consistent=bool(mesh.is_winding_consistent),degenerate_triangles=int(np.count_nonzero(mesh.area_faces<=0)),basic_topology_supported=topology,closest_descriptor_id=descriptors[int(order[0])]['Particle ID'] if raw_volume>0 else None,closest_relative_volume_difference_if_um=float(differences[order[0]]) if raw_volume>0 else None,second_relative_volume_difference_if_um=float(differences[order[1]]) if raw_volume>0 else None,repairs=[],coordinate_welding='Exact equality only; decoding duplicated STL coordinates')
        rows.append(row)
        print(member.filename,topology,row['closest_descriptor_id'],row['closest_relative_volume_difference_if_um'],flush=True)
matched=[r for r in rows if r['closest_descriptor_id'] is not None]
ids=[r['closest_descriptor_id'] for r in matched]
metadata=read_json(metadata_path)
unchanged=all(sha256(p)==v for p,v in inputs.items())
external=read_json(source_root/'local-source-receipt.json')['files']
external_unchanged=all(sha256(p)==v for p,v in external.items())
assert unchanged and external_unchanged
write_json(out/'report.json',dict(identity=out.name,input_sha256=inputs,inputs_unchanged=unchanged,external_inputs_unchanged=external_unchanged,archive_members=len(rows),descriptor_rows=len(descriptors),metadata_declared_particles=int(metadata['number_of_particles']),metadata_microns_per_pixel=metadata['microns_per_pixel'],unique_nearest_descriptor_assignment=len(set(ids))==len(ids),unmatched_descriptor_ids=sorted(set(d['Particle ID'] for d in descriptors)-set(ids),key=int),basic_topology_supported_count=sum(r['basic_topology_supported'] for r in rows),maximum_relative_volume_difference_if_um=max(r['closest_relative_volume_difference_if_um'] for r in matched),minimum_nearest_second_separation_if_um=min(r['second_relative_volume_difference_if_um']-r['closest_relative_volume_difference_if_um'] for r in matched),rows=rows,coordinate_units_admitted=False,prepared_library=False,material_registered=False,gpu_executed=False,scientific_admission=False,scope=__doc__,limitations=['Metadata, descriptor and mesh inventories are compared explicitly; no omitted grains are invented.','Nearest signed-volume matching is an observed diagnostic, not by itself an admitted scale or one-to-one source assignment.','Basic topology flags do not include full signed-distance, vertex-link, pseudonormal, inertia or quadrature qualification.','No mechanical parameters, grading, batch equivalence or experimental properties are inferred from the material name.']))
print('Completed',len(rows),'source members',flush=True)
