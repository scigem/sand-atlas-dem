"""Explicit fixed-frame drained observation replay; no fitting or preparation.

Measured radial effective stress is a constraint, not a claim about apparatus
control. All original observations remain ordered, including reversals/duplicates.
"""
import copy
import hashlib
import json
import math
import re
from .dsl import ValidationError
from .model import ROOT, load, physical_time, reject_duplicate_keys
from .recorded import validate_recorded_run
from .structure import check


def read_json(path):
    return parse_json(path.read_bytes())


def parse_json(content):
    return json.loads(content, object_pairs_hook=reject_duplicate_keys,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValidationError(f'invalid number: {s}')))


def digest(value):
    """Hash canonical JSON; file byte hashes are recorded separately by the CLI."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def prepare_replay(experiment, configuration, model=None):
    """Return request and conversion provenance without modifying input objects.

    Every non-stress state component and parameter is supplied explicitly.
    Material domain/initial-history consistency remains the engine's task.
    """
    check(experiment, json.loads((ROOT/'schemas/experiment.schema.json').read_text()))
    check(configuration, json.loads((ROOT/'schemas/drained-replay.schema.json').read_text()))
    if experiment['protocol'] != 'drained-triaxial':
        raise ValidationError('only normalized drained-triaxial experiments are supported')
    if not re.fullmatch(r'[a-z][a-z0-9-]*', configuration['model']):
        raise ValidationError('invalid model identifier')
    model = load(ROOT/'models'/f"{configuration['model']}.json") if model is None else model
    if model['id'] != configuration['model'] or physical_time(model):
        raise ValidationError('model must match and use incremental rather than physical-time execution')
    if model['state'][0]['name'] != 'sigma' or model['state'][0].get('shape') != 'sym':
        raise ValidationError('stress-first Mandel state required')
    decl = configuration['declarations']
    for key in ('evidence_url', 'evidence_locator', 'initialization_note'):
        if not decl[key].strip():
            raise ValidationError(f'{key}: non-whitespace declaration required')
    strain = configuration['initial_strain']
    if strain[1] != strain[2] or any(strain[3:]):
        raise ValidationError('initial strain must be axisymmetric with zero shear')
    params = configuration['parameters']
    if set(params) != set(model['parameters']):
        raise ValidationError('all and only named model parameters must be supplied')
    for name, field in model['parameters'].items():
        if not field['min'] <= params[name] <= field['max']:
            raise ValidationError(f'{name}: parameter outside declared bounds')
    history = configuration['initial_history']
    if set(history) != {field['name'] for field in model['state'][1:]}:
        raise ValidationError('all and only non-stress state fields must be supplied')
    flat_history = []
    for field in model['state'][1:]:
        value = history[field['name']]
        symmetric = field.get('shape', 'scalar') == 'sym'
        if symmetric != isinstance(value, list):
            raise ValidationError(f"{field['name']}: initial history shape mismatch")
        flat_history.extend(value if symmetric else [value])
    source_e = experiment['initial'].get('void_ratio')
    override = configuration.get('initial_void_ratio_override_note')
    if override is not None and not override.strip():
        raise ValidationError('void-ratio override requires a non-whitespace explanation')
    if source_e is not None:
        if type(source_e) not in (int, float) or not math.isfinite(source_e):
            raise ValidationError('source initial void ratio must be finite')
        if 'e' in history and (isinstance(history['e'], list)
                              or abs(history['e']-source_e) > 1e-10*max(1, abs(source_e))):
            if override is None:
                raise ValidationError('initial e differs from source void ratio; explicit override note required')
    rows = experiment['rows']
    if not 2 <= len(rows) <= 5001:
        raise ValidationError('2–5001 observations required; no automatic downsampling')
    if any('p' not in row for row in rows):
        raise ValidationError('every observation requires mean effective stress p')
    first = rows[0]
    radial = lambda row: row['p'] - row['q']/3
    sigma = [first['p']+2*first['q']/3, radial(first), radial(first), 0, 0, 0]
    if not all(math.isfinite(v) for v in sigma):
        raise ValidationError('derived initial stress is not finite')
    # A second initial stress declaration must not silently contradict row zero.
    known = experiment['initial'].get('effective_stress')
    if known is not None:
        if (not isinstance(known, list) or len(known) != 6
                or any(type(a) not in (int, float) or not math.isfinite(a)
                       or abs(a-b) > 1e-8*max(1, abs(b)) for a,b in zip(known,sigma))):
            raise ValidationError('experiment initial effective stress contradicts first observation')
    controls = [{'kind':'stress' if i in (1,2) else 'strain',
                 'coefficients':[int(i == j) for j in range(6)]} for i in range(6)]
    knots = [{'coordinate':i, 'targets':[strain[0]+(row['axial_strain']-first['axial_strain']),
                                         radial(row), radial(row), 0, 0, 0]}
             for i,row in enumerate(rows)]
    request = {'version':'experimental-recorded-run-0.1', 'model':model['id'],
               'parameters':[params[name] for name in model['parameters']],
               'initial':sigma+flat_history,
               'path':{'version':'experimental-recorded-path-0.1',
                       'coordinate':{'name':'source_row_index','unit':'1','meaning':'path_order'},
                       'initial_strain':copy.deepcopy(strain), 'controls':controls, 'knots':knots}}
    for key in ('integration_settings','control_settings','restart_settings'):
        if key in configuration:
            request[key] = copy.deepcopy(configuration[key])
    validate_recorded_run(request)
    provenance = {'version':'experimental-drained-replay-provenance-0.1',
                  'experiment_id':experiment['id'], 'experiment_sha256':digest(experiment),
                  'configuration_sha256':digest(configuration), 'model_sha256':digest(model),
                  'request_sha256':digest(request), 'source':copy.deepcopy(experiment['provenance']),
                  'declarations':copy.deepcopy(decl), 'observation_count':len(rows),
                  'row_mapping':'knot i corresponds to input rows[i], zero-based; no filtering',
                  'source_axial_strain_origin':first['axial_strain'],
                  'initial_void_ratio_comparison':{'source':source_e, 'supplied_e':history.get('e'),
                                                   'override_note':override},
                  'limitations':[
                      'Measured radial effective stress is replayed; this is not an apparatus-control classification.',
                      'Coordinate is observation order, not elapsed time.',
                      'All history is caller supplied; no preparation or calibration is inferred.',
                      'q enters the radial constraint through p-q/3; its curve is not an independent input channel.',
                      'Conversion does not establish model applicability, source agreement or redistribution permission.']}
    return {'request':request, 'provenance':provenance}
