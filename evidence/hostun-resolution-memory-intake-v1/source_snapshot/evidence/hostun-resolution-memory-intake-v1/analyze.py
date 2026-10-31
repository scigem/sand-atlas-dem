"""Metadata-only planning bounds; not a measured peak or device allocation."""
import ast
import hashlib
import json
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ASSET = ROOT / 'evidence/hostun-adaptive-l5-candidate-v1/private/adaptive.h5'
RECEIPT = ROOT / 'evidence/hostun-adaptive-l5-candidate-v1/verification.json'
ENGINE = ROOT / 'src/sand_atlas_dem/gpu.py'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

paths = [Path(__file__), ENGINE, RECEIPT]
frozen = {str(p.relative_to(ROOT)): digest(p) for p in paths}
assert not (OUT / 'prospective-freeze.json').exists()
(OUT / 'prospective-freeze.json').write_text(json.dumps(dict(
    files=frozen, asset_sha256=json.loads(RECEIPT.read_text())['asset_sha256'],
    scope=__doc__, acceptance='No scientific acceptance gate. Evaluate the unchanged Engine buffer estimate at declared capacities and both allowed bin-count bounds. Plus-one quadrature is hypothetical, not generated or qualified.'
), indent=2) + '\n')
assert digest(ASSET) == json.loads(RECEIPT.read_text())['asset_sha256']
# Evaluate the actual frozen Engine estimate AST, rather than duplicate constants.
tree = ast.parse(ENGINE.read_text())
engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Engine')
init = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
start = next(i for i,n in enumerate(init.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'estimated' for t in n.targets))
end = next(i for i,n in enumerate(init.body) if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) and n.test.left.id == 'estimated')
module = ast.Module(body=init.body[start:end], type_ignores=[])
code = compile(ast.fix_missing_locations(module), str(ENGINE), 'exec')
from types import SimpleNamespace
with h5py.File(ASSET) as f:
    physical = [(len(g['faces']), len(g['samples']), g['sdf'].size * g['sdf'].dtype.itemsize) for g in f.values()]
assert len(physical) == 53
rows = []
for profile,factor in [('revised-candidate',1),('hypothetical-plus-one',4)]:
    library = SimpleNamespace(shapes=[SimpleNamespace(faces=range(nf),samples=range(ns*factor),sdf=SimpleNamespace(nbytes=nb)) for nf,ns,nb in physical])
    for particles in (424,10000,50000):
        estimates=[]
        for bins in (1,64**3):
            contacts=max(16,particles*4096)
            state=SimpleNamespace(contact_capacity=contacts,hash_size=1 << (max(32,2*contacts)-1).bit_length(),pair_capacity=max(1,particles*32),bin_count=bins,bin_capacity=128,n=particles,wall_count=0)
            namespace=dict(self=state,spec=SimpleNamespace(contact_geometry='mesh'),library=library)
            exec(code,namespace)
            estimates.append(namespace['estimated'])
        rows.append(dict(profile=profile,particles=particles,contacts_per_particle=4096,pairs_per_particle=32,bin_capacity=128,minimum_estimate_bytes=estimates[0],maximum_estimate_bytes=estimates[1],minimum_estimate_GiB=estimates[0]/1024**3,maximum_estimate_GiB=estimates[1]/1024**3,within_12GiB_for_all_bin_counts=estimates[1]<=12*1024**3))
assert all(digest(ROOT / name)==value for name,value in frozen.items())
report=dict(identity=OUT.name,source_unchanged=True,metadata_only=True,gpu_executed=False,scientific_admission=False,shapes=53,candidate_nodes=sum(ns for _,ns,_ in physical),rows=rows,limitations=['Engine estimates are not measured total application or driver memory. Host copies, compilation, rollback and output peaks require separate measurement.','4096 contacts per particle and 32 pairs per particle retain the current 424-case capacities; they are not proven adequate for dense 10k/50k cases.','Bin-count extremes bound the existing 1..64 per-axis implementation, not an observed specimen.','Hypothetical plus-one multiplies each face sample count by four; no asset or numerical qualification is claimed.','Memory fit does not establish stable timestep, usable throughput or physical convergence.'])
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
for r in rows:
    print(r['profile'],r['particles'],round(r['minimum_estimate_GiB'],3),round(r['maximum_estimate_GiB'],3),'GiB')
