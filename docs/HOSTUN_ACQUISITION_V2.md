# Hostun source acquisition and scale crosswalk — revision 2

2026-10-31. This report supersedes the availability/scale blocker in
HOSTUN_SOURCE_GAP.md; the original rejection and failure log remain intact.
Scientific admission of the full material study remains open.

## Sources and exact identities

- Source landing page: https://sand-atlas.scigem.com/sands/hostun-0000005/
- Archive: https://data.scigem-eng.sydney.edu.au/sand-atlas/sands/hostun-0000005/meshes_30.zip
  SHA-256 `22549ef78526991dc3adb82efe7f1b9cf98babef2e00b435c449f3b90d0f48a6`.
- Dimensioned descriptor table: https://sand-atlas.scigem.com/assets/sands/hostun-0000005.csv
  SHA-256 `3a615e1af45b981a8c39da454155fe37dd7e61d422d6b91c13fa68af283f5872`.
- Metadata source: adjacent `sand-atlas/_data/json/hostun-0000005.json`, contributor
  Max Wiebicke, Hostun angular quartz, nominal voxel size 1 µm, original publication
  https://doi.org/10.1088/1361-6501/aa8dbf, advertised source rights CC BY 4.0.

Original archive, descriptors and all derived grain assets remain under ignored
`assets/private/`. Nothing has been uploaded or redistributed. The exact private
preprocessing command is `python examples/prepare_hostun.py 32 512 4`, followed by
`64 2048 4` for the refinement library. It never executes source-supplied code.

## What changed the evidence

The API bulk endpoint failure did not establish absence of the dataset. The website's
direct static archive link succeeded. It contains **54** STL meshes, and the descriptor
table contains 54 rows, although the landing page advertises 53. A mesh filename's
numeric suffix is not its descriptor ID. Sorting and matching physical volumes gives
a unique one-to-one correspondence for all 54 records, retained explicitly in
`evidence/hostun-ingestion-r32-s512-v1.json`.

The CSV labels volume in µm³ and equivalent diameter in µm. Raw mesh coordinate-volume
values agree with those dimensioned source volumes: most residuals are below 0.01%;
the largest residual is about 0.48%. This is strong dimensional evidence for converting
raw coordinates as **µm**, with a 1e-6 factor to metres. It is an explicit inference
from source descriptors, not an STL-embedded unit declaration or an assumed plausible
sand diameter. Each shape records that inference and its descriptor identity.
The current processing package's mm export convention is not used for these assets.

No volume-match acceptance tolerance was retrofitted to observed results: this was a
source identity/unit discovery crosswalk, with all residuals retained. It is not a
convergence or experimental validation test. The two outlying residuals and the
54-versus-53 metadata discrepancy remain visible for source review.

## Geometry decisions and outputs

Exact-coordinate welding decodes triangle-soup STL. No smoothing, hole filling,
winding changes or component deletion occurs. Of 54 meshes, 53 are single closed,
consistently wound positive-volume solids and pass the strict importer. The remaining
`particle_00019.stl` is byte-identical to the originally rejected reference mesh;
it retains both components and is excluded as a whole. This selects a documented
53-grain subset; numerical agreement with the advertised count does not establish
that it is the website authors' intended selection.

The complete topology inventory is `evidence/hostun-archive-topology-v2.json`.
The first library is `assets/private/hostun-30-r32-s512.h5`, SHA-256
`7757ab752483cd8fabd910ea2cba3c5570fa8a6962b7672d14a32b15f8c414a9`.
Every source grain has separate SHA-256, descriptor mapping, original physical volume,
SDF dimensions/spacing and output hash in the ingestion report. Per-grain cache files
allow interrupted preprocessing to continue without altering accepted source bytes.

The first library's mean equivalent diameter is approximately 345 µm. Uniform reuse
of these shapes in a 10k assembly would be a declared synthetic sampling policy,
not a measured specimen's grain-size distribution or preparation history.
Shape/SDF/contact convergence, prepared specimen qualification and experimental
validation remain separate requirements.
