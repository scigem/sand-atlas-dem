# Hostun experimental comparison admission

The first read-only experimental intake is retained in
`evidence/hostun-experimental-intake-v1/report.json`, with the exact script and
hashes of all external inputs. It reads the adjacent Atlas without modifying it.
This is source intake, not an experimental validation or a calibrated material.

Ten existing curated drained-triaxial Hostun records are available from GEOLAB v2
(DOI 10.5281/zenodo.12697903). Their native channel units, source-cell identities,
row order and warnings are preserved. None declares a normalized initial state;
all retain source-defined signs/conventions. Repeated and decreasing native
abscissae remain visible. These observations do not establish whether a reversal
is physical, digitization-related or a table-order issue. No sorting, averaging,
interpolation, duplicate removal or cross-curve joining is performed.

The existing adjacent Atlas packets `hostun_feasibility.json` and
`hostun_source_metadata.json` are read-only inputs with receipt hashes. Their
prior review identifies conflicts between workbook relative-density labels and
original source tables; the Deng-cited original figure labels shear strain where
the worksheet labels axial strain. The original specimen state, strain definition
and curve correspondence must be resolved before translating any such record
into an imposed DEM or continuum loading path. The present intake does not repeat
or override that source review.

The Sand Atlas geometry metadata cites DOI 10.1088/1361-6501/aa8dbf. The experimental
records have a different provenance route through GEOLAB and its original test
references. Sharing the name Hostun does not establish identical batches, grading,
preparation or specimens. The balanced 53-shape numerical inventory is also not an
inferred full-material particle-size distribution.

| Requirement | Current evidence and next action |
| --- | --- |
| Geometry identity and SI scale | Established for the documented subset; retain its exclusions and physical asset hash |
| Experimental batch/grading identity | Not established against that subset; acquire source material and grading correspondence |
| Axial/shear strain and signs | Source conflict remains; resolve definitions against original test documentation |
| Initial confinement and void ratio | Raw metadata exist but normalized specimen states are unadmitted; resolve state timing and curve mapping |
| Volume evolution | A peak dilation angle is not a volumetric-strain history; do not fabricate one |
| DEM mechanical calibration | Example density/stiffness/friction are explicit software inputs, not a calibration; no fitting has run |
| Preparation/loading qualification | Bounded 424-grain preparation is running; mesh work, timestep, resolution and loading-rate studies remain required |
| Calibration versus validation | Freeze distinct source-supported specimen assignments and metrics before fitting; no split is assigned by this intake |
| Continuum comparison | Native replay software is verified; actual qualified DEM/experimental comparison remains open |

The next experimental action is resolution of source state/strain/batch
correspondence, followed by a prospective calibration/validation protocol. A
stress–strain overlay, a successful solver run or a small objective value cannot
supply missing source identity. Keep source acquisition, numerical verification,
calibration and independent validation separately reported.
