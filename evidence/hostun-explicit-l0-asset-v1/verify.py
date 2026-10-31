"""Check rebuilt explicit L0 asset against every array of the retained old L0 asset."""
from pathlib import Path
import numpy as np
from sand_atlas_dem.geometry import ShapeLibrary
from sand_atlas_dem.util import sha256, write_json
root = Path(__file__).resolve().parent
old = Path('/Users/bmar5496/code/sand-atlas-dem/assets/private/hostun-30-r64-gauss3.h5')
new = root / 'private/coarse.h5'
hashes = {str(p): sha256(p) for p in (old, new, Path(__file__))}
a, b = ShapeLibrary.load(old), ShapeLibrary.load(new)
assert len(a.shapes) == len(b.shapes) == 53
for x, y in zip(a.shapes, b.shapes):
    assert x.id == y.id
    assert y.provenance['quadrature_subdivisions'] == 0
    for field in ('vertices', 'faces', 'samples', 'areas', 'sdf', 'origin', 'inertia'):
        assert np.array_equal(getattr(x, field), getattr(y, field)), (x.id, field)
    for field in ('volume', 'radius', 'spacing'):
        assert np.array_equal(getattr(x, field), getattr(y, field)), (x.id, field)
assert all(sha256(p) == h for p, h in hashes.items())
write_json(root / 'verification.json', dict(passed=True, source_unchanged=True,
           all_53_shapes_exact_array_agreement=True, hashes=hashes,
           scope='Only explicit L0 provenance differs. No force, assembly or material admission.'))
