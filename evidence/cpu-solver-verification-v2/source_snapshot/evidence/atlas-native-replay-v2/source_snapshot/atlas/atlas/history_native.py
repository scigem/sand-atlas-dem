"""Explicit existing-native callback for the experimental legacy snapshot bridge.

No compilation or preparation. The full existing integrate endpoint receives the
original parent, including reset memory; its settings response binds every input.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

from .dsl import ValidationError
from .initialization import canonical_hash
from .model import reject_duplicate_keys

SETTING_BOUNDS = {'residual_tolerance': (0, None), 'rate_tolerance': (0, None),
                  'max_iterations': (1, 1000), 'max_depth': (0, 20),
                  'max_substeps': (1, 1000000), 'max_backtracks': (0, 30)}


def _read_json(data):
    return json.loads(data, object_pairs_hook=reject_duplicate_keys,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValidationError('nonfinite native JSON')))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _echo(actual, expected):
    if type(expected) in (int,float):
        try:
            return type(actual) in (int,float) and math.isfinite(actual) and actual == expected
        except OverflowError:
            return False
    if isinstance(expected,list):
        return isinstance(actual,list) and len(actual)==len(expected) and all(_echo(a,b) for a,b in zip(actual,expected))
    if isinstance(expected,dict):
        return isinstance(actual,dict) and set(actual)==set(expected) and all(_echo(actual[k],v) for k,v in expected.items())
    return type(actual) is type(expected) and actual == expected


def native_integrator(solver, catalogue, settings, *, timeout=30):
    """Bind an existing binary/catalogue/settings; return a full-step callback.

    All six settings are mandatory. The generated catalogue must match its own
    canonical hash, runtime --describe, wire parameter order and requested model bytes. Each response must echo the
    catalogue identity, complete request and resolved settings. Binary identity
    is checked before and after execution and included with returned diagnostics.
    """
    if not isinstance(settings, dict) or set(settings) != set(SETTING_BOUNDS):
        raise ValidationError('all and only six native integration settings required')
    for name, (lower, upper) in SETTING_BOUNDS.items():
        value = settings[name]
        integer = name not in ('residual_tolerance', 'rate_tolerance')
        try:
            valid = (type(value) is int if integer else type(value) in (int,float)) and math.isfinite(value)
            valid = valid and (value >= lower if integer else value > lower) and (upper is None or value <= upper)
        except OverflowError:
            valid = False
        if not valid: raise ValidationError('invalid native integration setting: '+name)
    try:
        timeout_valid = type(timeout) in (int,float) and math.isfinite(timeout) and 0 < timeout <= 60
    except OverflowError:
        timeout_valid = False
    if not timeout_valid:
        raise ValidationError('native timeout must be finite and in (0,60] seconds')
    settings = copy.deepcopy(settings)
    resolved_settings = {'version':'incremental-settings-0.1','norm':'scaled_component_max',**settings}
    solver = Path(solver).resolve()
    data = _read_json(Path(catalogue).read_bytes())
    if (set(data) != {'schema_version','sha256','models'} or data['schema_version'] != '0.1'
            or not isinstance(data['models'], list) or canonical_hash(data['models']) != data['sha256']):
        raise ValidationError('native generated catalogue hash or format mismatch')
    models = {m['id']: canonical_hash(m) for m in data['models']}
    if len(models) != len(data['models']): raise ValidationError('duplicate native catalogue model identity')
    parameter_orders = {m['id']: list(m['parameters']) for m in data['models']}
    binary_hash = _sha(solver)
    described = subprocess.run([str(solver),'--describe'],capture_output=True,text=True,timeout=timeout)
    if _sha(solver) != binary_hash or described.returncode != 0:
        raise ValidationError('native describe failed or binary identity changed')
    compiled = _read_json(described.stdout)
    if not _echo(compiled,data):
        raise ValidationError('native compiled catalogue differs from supplied catalogue')
    compiled_orders = {m['id']:list(m['parameters']) for m in compiled['models']}
    if compiled_orders != parameter_orders:
        raise ValidationError('native compiled parameter order differs from supplied catalogue')

    def integrate(model, state, increment, parameters):
        if (models.get(model['id']) != canonical_hash(model)
                or parameter_orders.get(model['id']) != list(model['parameters'])):
            raise ValidationError('native compiled model differs from snapshot model')
        if _sha(solver) != binary_hash: raise ValidationError('native binary changed since callback binding')
        flat = [v for f in model['state'] for v in (state[f['name']] if f.get('shape')=='sym' else [state[f['name']]])]
        params = [parameters[name] for name in model['parameters']]
        csv = lambda xs: ','.join(repr(float(x)) for x in xs)
        command = [str(solver), '--mode', 'integrate', '--model', model['id'],
                   '--parameters', csv(params), '--initial', csv(flat), '--strain', csv(increment),
                   '--integration-settings', csv([settings[key] for key in SETTING_BOUNDS])]
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if _sha(solver) != binary_hash: raise ValidationError('native binary changed during callback')
        response = _read_json(process.stdout)
        if not isinstance(response,dict) or type(response.get('ok')) is not bool or process.returncode != (0 if response['ok'] else 1):
            raise ValidationError('native integration status/exit mismatch')
        for key, expected in {'catalogue_hash': data['sha256'], 'model': model['id'],
                              'parameters': params, 'initial': flat, 'strain': increment,
                              'integration_settings': resolved_settings}.items():
            if not _echo(response.get(key),expected): raise ValidationError('native integration echo mismatch: '+key)
        raw = response.get('state')
        if (not isinstance(raw,list) or len(raw) != len(flat)
                or any(type(x) not in (int,float) or not math.isfinite(x) for x in raw)):
            raise ValidationError('invalid native integration state')
        named = {}; offset = 0
        for field in model['state']:
            size = 6 if field.get('shape') == 'sym' else 1
            named[field['name']] = raw[offset:offset+size] if size == 6 else raw[offset]
            offset += size
        diagnostics = copy.deepcopy(response.get('diagnostics'))
        if not isinstance(diagnostics,dict): raise ValidationError('missing native integration diagnostics')
        for key in ('iterations','substeps','rejected'):
            if type(diagnostics.get(key)) is not int or diagnostics[key] < 0:
                raise ValidationError('invalid native integration diagnostic counter')
        diagnostics['native_runtime'] = {'binary_sha256': binary_hash, 'catalogue_sha256': data['sha256'],
                                         'settings': copy.deepcopy(resolved_settings), 'model_sha256': models[model['id']],
                                         'request_sha256': canonical_hash({'model': model['id'], 'initial': flat,
                                                                          'parameters': params, 'increment': increment,
                                                                          'settings': settings})}
        return {'ok': response['ok'], 'state': named, 'diagnostics': diagnostics,
                'message': response.get('message','')}
    return integrate
