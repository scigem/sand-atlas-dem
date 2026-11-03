"""Auditable public measurements, independent of optional source-format readers.

The numerical core and website build do not depend on HDF5 or spreadsheet tools.
Source readers produce plain Python values; curated records retain every channel
and the browser build selects only the channels it can currently plot.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import re


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                       allow_nan=False) + '\n').encode()


# source name: (Atlas name, expected source unit, SI multiplier)
OPARA_CHANNELS = {
    'time': ('elapsed_time', 's', 1),
    'axial_strain': ('axial_strain', '-', 1),
    'volumetric_strain': ('volumetric_strain', '-', 1),
    'deviatoric_stress': ('q', 'kPa', 1000),
    'mean_effective_stress': ('p', 'kPa', 1000),
    'effective_axial_stress': ('axial_effective_stress', 'kPa', 1000),
    'effective_radial_stress': ('radial_effective_stress', 'kPa', 1000),
    'pore_water_pressure': ('pore_pressure_absolute', 'kPa', 1000),
    'cell_pressure': ('cell_pressure', 'kPa', 1000),
    'total_axial_stress': ('total_axial_stress', 'kPa', 1000),
    'axial_displacement': ('axial_displacement', 'mm', .001),
    'axial_force': ('axial_force', 'kN', 1000),
    'cross-sectional_area': ('cross_sectional_area', 'cm^2', .0001),
    'volume_change': ('volume_change', 'cm^3', .000001),
    'void_ratio': ('void_ratio', '-', 1),
}


def normalize_opara(channels, channel_attributes, inputs, file_attributes, provenance):
    """Normalize a measured OPARA shearing phase without interpolation or sampling."""
    if file_attributes.get('virtual') is not False:
        raise ValueError('Expected measured, non-virtual OPARA data')
    protocol = {
        'analysis/triaxial_test/consolidated_drained': 'drained-triaxial',
        'analysis/triaxial_test/consolidated_undrained': 'undrained-triaxial',
    }.get(file_attributes.get('schema'))
    if not protocol:
        raise ValueError('Unrecognized OPARA test schema')
    required = set(OPARA_CHANNELS) - {'void_ratio'}
    if not required <= channels.keys() or set(channels) - OPARA_CHANNELS.keys():
        raise ValueError('OPARA channels differ from the reviewed source layout')
    length = len(channels['time'])
    if not 1 <= length <= 100000:
        raise ValueError('Invalid measurement count')
    rows = [{} for _ in range(length)]
    mapping = {}
    for source, values in channels.items():
        name, unit, factor = OPARA_CHANNELS[source]
        if channel_attributes[source].get('unit') != unit:
            raise ValueError('Unexpected source unit for ' + source)
        if len(values) != length or not all(isinstance(x, (int, float)) and
                                           math.isfinite(x) for x in values):
            raise ValueError('Invalid or unaligned source channel: ' + source)
        mapping[source] = dict(channel=name, source_unit=unit, multiplier=factor)
        for row, value in zip(rows, values):
            row[name] = value * factor
    times = [row['elapsed_time'] for row in rows]
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError('Elapsed time is not in acquisition order')
    for name, unit in [('initial_void_ratio', '-'), ('effective_confining_pressure', 'kPa')]:
        item = inputs[name]
        if item['attributes'].get('unit') != unit or not math.isfinite(item['value']):
            raise ValueError('Invalid initial state: ' + name)
    if inputs['initial_void_ratio']['value'] <= 0:
        raise ValueError('Invalid initial void ratio')
    # Difference is computed in source kPa before scaling, matching the import
    # contract used for other measured pore-pressure changes.
    baseline = channels['pore_water_pressure'][0]
    for row, value in zip(rows, channels['pore_water_pressure']):
        row['pore_pressure'] = (value - baseline) * 1000
    first = rows[0]
    provenance = dict(provenance, source_attributes=file_attributes,
                      source_channel_attributes=channel_attributes,
                      source_inputs=inputs, channel_mapping=mapping,
                      transformations=[
                          'Stress kPa to Pa; force kN to N; length mm to m; area cm² to m²; volume cm³ to m³.',
                          'Source strains and void ratios are dimensionless and retained without percent conversion.',
                          'Pore-pressure change = measured absolute pore pressure minus its first reading, in Pa.',
                          'Author-supplied membrane-corrected stresses retained; no rederivation, resampling or rounding.',
                      ])
    return dict(schema_version='0.1', kind='experiment', id=provenance.pop('test_id'),
                material='Clayey-silty sand (Ullmann et al.)', protocol=protocol,
                conventions=dict(stress='compression-positive', strain='compression-positive',
                                 stress_unit='Pa', strain_unit='1'),
                initial=dict(effective_stress=[first['axial_effective_stress'],
                                              first['radial_effective_stress'],
                                              first['radial_effective_stress'], 0, 0, 0],
                             void_ratio=inputs['initial_void_ratio']['value'],
                             nominal_confinement=inputs['effective_confining_pressure']['value'] * 1000),
                provenance=provenance,
                warnings=['Shearing phase only; consolidation history is not included.',
                          'Measured initial stress can differ from nominal consolidation pressure.',
                          'Post-localization specimen averages are not homogeneous material-point measurements.'],
                rows=rows)


def load_curated(root):
    """Verify pinned assets and their source-specific release terms before use."""
    root = Path(root)
    index = root / 'data/curated/index.json'
    if not index.exists():
        return []
    registry = json.loads((root / 'data/public_sources.json').read_text())
    sources = {s['id']: s for s in registry['sources']}
    manifest = json.loads(index.read_text())
    if manifest.get('format') != 'atlas-curated-experiments-v1':
        raise ValueError('Unrecognized curated manifest')
    records, ids = [], set()
    for entry in manifest['records']:
        name = entry['path']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*\.json\.gz', name):
            raise ValueError('Invalid curated path')
        content = (index.parent / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError('Curated checksum mismatch: ' + name)
        record = json.loads(gzip.decompress(content))
        kind = record.get('kind')
        if kind not in ('experiment', 'measurement-series', 'test-summary') or record.get('schema_version') != '0.1':
            raise ValueError('Unsupported curated record schema')
        expected = (dict(stress='compression-positive', strain='compression-positive', stress_unit='Pa', strain_unit='1')
                    if kind == 'experiment' else
                    dict(stress='source-defined', strain='source-defined', stress_unit='per-channel', strain_unit='per-channel'))
        if record.get('conventions') != expected:
            raise ValueError('Invalid curated conventions')
        if not 1 <= len(record['rows']) <= (100000 if kind == 'experiment' else 200000):
            raise ValueError('Invalid curated observation count')
        allowed = None
        if kind != 'experiment':
            allowed = {c['key'] for c in record['columns']}
            if (len(allowed) != len(record['columns']) or not 2 <= len(allowed) <= 100 or
                any(not re.fullmatch(r'c[0-9]+', c['key']) or not isinstance(c.get('name'), str) or
                    not c['name'] or not isinstance(c.get('unit'), str) or not c['unit']
                    for c in record['columns'])):
                raise ValueError('Invalid source-series columns')
            if set(record['source_plot']) != {'x', 'y'} or not set(record['source_plot'].values()) <= allowed:
                raise ValueError('Invalid source-series axes')
        for row in record['rows']:
            if ((kind == 'experiment' and not {'axial_strain', 'q'} <= row.keys()) or
                (allowed is not None and (not row or not row.keys() <= allowed)) or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in row.values())):
                raise ValueError('Invalid curated observation')
        provenance = record['provenance']
        source = sources[entry['source_id']]
        if (not source.get('public_hosting') or not source.get('attribution') or
                not source.get('license_url') or not source.get('license_evidence')):
            raise ValueError('Missing source release terms')
        if provenance['archive_sha256'] not in {f['sha256'] for f in source['files']}:
            raise ValueError('Record is not from a pinned public source')
        for key in ('source_url', 'license', 'license_url', 'attribution'):
            if provenance[key] != source[key]:
                raise ValueError('Record release terms differ from public source registry')
        if record['id'] != entry['id'] or len(record['rows']) != entry['rows'] or record['id'] in ids:
            raise ValueError('Curated record identity/count mismatch')
        json_bytes(record)  # Reject non-finite values even in hand-edited assets.
        ids.add(record['id'])
        records.append(record)
    return records
