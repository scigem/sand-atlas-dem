"""Immutable Wichtmann archive ingestion with explicit, inspectable mappings."""
from __future__ import annotations
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

CONVENTIONS={'stress':'compression-positive','strain':'compression-positive','stress_unit':'Pa','strain_unit':'1'}
SOURCES={
 'TMD':'https://www.torsten-wichtmann.de/KFSDB-TMD.htm',
 'TMU':'https://www.torsten-wichtmann.de/KFSDB-TMU.htm',
}


def numeric_rows(text):
    """Two header lines plus blank lines; reject malformed numeric records."""
    rows=[]
    lines=text.splitlines()
    if len(lines)<4:raise ValueError('file is missing headers or data')
    for number,line in enumerate(lines[2:],3):
        if not line.strip():continue
        try:row=[float(v) for v in line.split()]
        except ValueError as exc:raise ValueError(f'non-numeric record at line {number}') from exc
        if len(row)!=8 or not all(math.isfinite(v) for v in row):raise ValueError(f'expected eight finite values at line {number}')
        rows.append(row)
    if not rows:raise ValueError('no data rows')
    return lines[:2],rows


def normalize_member(content,name,archive_hash,member_hash,metadata=None):
    match=re.fullmatch(r'(TMD|TMU)(\d+)\.dat',Path(name).name,re.IGNORECASE)
    if not match:raise ValueError(f'unsupported test file: {name}')
    series=match[1].upper();test_id=series+match[2]
    headers,raw=numeric_rows(content.decode('utf-8-sig'))
    header=headers[0].split()
    if series=='TMD' and header[:4]!=['eps1','epsv','eps3','epsq']:raise ValueError('unrecognized TMD column layout')
    if series=='TMU' and header[:4]!=['eps1','u','sigma3',"sigma3'"]:raise ValueError('unrecognized TMU column layout')
    rows=[]
    for r in raw:
        if series=='TMD':
            rows.append(dict(axial_strain=r[0]/100,volumetric_strain=r[1]/100,radial_strain=r[2]/100,void_ratio=r[4],q=r[5]*1000,p=r[6]*1000))
        else:
            rows.append(dict(axial_strain=r[0]/100,pore_pressure=(r[1]-raw[0][1])*1000,
                             pore_pressure_absolute=r[1]*1000,radial_effective_stress=r[3]*1000,
                             axial_effective_stress=r[5]*1000,p=r[6]*1000,q=r[7]*1000))
    known=dict(metadata or {})
    if series=='TMD':known['void_ratio']=rows[0]['void_ratio']
    first=rows[0];known.update(effective_stress=[first['p']+2*first['q']/3,first['p']-first['q']/3,first['p']-first['q']/3,0,0,0])
    warnings=['Source column labelled Void ratio [%] is interpreted as dimensionless e for TMD; original values retained.' ] if series=='TMD' else ['Zero volumetric strain is a protocol assumption, not a measured channel; it is not inserted into rows.']
    if 'void_ratio' not in known:warnings.append('Initial void ratio must be extracted from the paper before constitutive comparison.')
    return dict(schema_version='0.1',kind='experiment',id='KFS-'+test_id,material='Karlsruhe fine sand',
                protocol='drained-triaxial' if series=='TMD' else 'undrained-triaxial',conventions=CONVENTIONS,
                provenance=dict(source_url=SOURCES[series],archive_sha256=archive_hash,member=name,member_sha256=member_hash,
                                original_headers=headers,normalized_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                redistribution='not-reviewed; local research copy only'),
                initial=known,warnings=warnings,rows=rows)


def ingest(archive,destination,metadata=None):
    archive=Path(archive);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest();outputs=[]
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():continue
            # Never extract paths, and reject decompression bombs before reading.
            if info.file_size>50_000_000:raise ValueError('archive member exceeds 50 MB')
            if not re.fullmatch(r'(TMD|TMU)\d+\.dat',Path(info.filename).name,re.IGNORECASE):continue
            b=z.read(info);test_id=Path(info.filename).stem.upper()
            record=normalize_member(b,info.filename,digest,hashlib.sha256(b).hexdigest(),(metadata or {}).get(test_id))
            out=destination/(record['id']+'.json');text=json.dumps(record,indent=2,allow_nan=False)+'\n'
            if out.exists():
                previous=json.loads(out.read_text())
                if previous['provenance']['member_sha256']!=record['provenance']['member_sha256']:raise ValueError(f'raw content changed for {out.name}; use a new data version')
                # Preserve the original normalization timestamp for repeatable ingestion.
                record['provenance']['normalized_at']=previous['provenance']['normalized_at'];text=json.dumps(record,indent=2,allow_nan=False)+'\n'
            out.write_text(text);outputs.append(out)
    if not outputs:raise ValueError('archive contains no supported TMD/TMU test files')
    return outputs
