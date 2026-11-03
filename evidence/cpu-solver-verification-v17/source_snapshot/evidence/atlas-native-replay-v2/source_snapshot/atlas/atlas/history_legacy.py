"""Experimental immutable transport around a trusted full legacy integrator.

No reset expressions are translated and no material equations are evaluated here.
The callback owns all domain/initial checks, preparation, _n semantics, integration
and cutbacks. This bridge validates identity/transport, not scientific correctness.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from .dsl import ValidationError
from .history import MAX_COUNTER, make_snapshot
from .model import validate

VERSION = 'experimental-legacy-history-snapshot-0.1'
_OUTER_KEYS = {'version', 'binding', 'history', 'strain', 'strain_sha256'}
_BINDING_KEYS = {'model_id', 'model_sha256', 'parameters', 'parameters_sha256', 'parameter_order', 'parameter_order_sha256', 'layout', 'layout_sha256'}


def _hash(value):
    try:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode()).hexdigest()
    except (ValueError, TypeError, ArithmeticError, RecursionError) as exc:
        raise ValidationError('legacy transport requires finite JSON data') from exc


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _six(values, label):
    if not isinstance(values, list) or len(values) != 6 or not all(_number(x) for x in values):
        raise ValidationError(label + ' requires six finite Mandel components')


def _model(model):
    try:
        validate(model)
        if model['schema_version'] != '0.1':
            raise ValidationError('legacy history bridge requires model schema 0.1')
        _hash(model)
    except (ValueError, TypeError, ArithmeticError, RecursionError) as exc:
        raise ValidationError('invalid legacy transport model: ' + str(exc)) from exc


def _parameters(model, parameters):
    if not isinstance(parameters, dict) or set(parameters) != set(model['parameters']):
        raise ValidationError('complete exact named legacy parameters required')
    resolved = {}
    for name, field in model['parameters'].items():
        value = parameters[name]
        if not _number(value) or not field['min'] <= value <= field['max']:
            raise ValidationError('invalid legacy parameter: ' + name)
        resolved[name] = float(value)
    return resolved


def _binding(model, parameters):
    parameters = _parameters(model, parameters)
    offset = 0; layout = []
    for index, field in enumerate(model['state']):
        size = 6 if field['shape'] == 'sym' else 1
        layout.append({'name': field['name'], 'slot': 'f' + str(index),
                       'shape': field['shape'], 'unit': copy.deepcopy(field['unit']),
                       'offset': offset, 'size': size,
                       'transient': field.get('transient', False),
                       'multiplier': field.get('multiplier', False)})
        offset += size
    return {'model_id': model['id'], 'model_sha256': _hash(model),
            'parameters': parameters, 'parameters_sha256': _hash(parameters),
            'parameter_order': list(model['parameters']),
            'parameter_order_sha256': _hash(list(model['parameters'])),
            'layout': layout, 'layout_sha256': _hash(layout)}


def _program(binding):
    # Source locator binds actual content identities, not merely a model title.
    identity = '; '.join(key + '=' + binding[key] for key in
                         ('model_sha256', 'parameters_sha256', 'parameter_order_sha256', 'layout_sha256'))
    return {'schema_version': 'experimental-history-0.1', 'id': 'legacy-state-transport',
            'sources': [{'url': 'https://github.com/benjym/constitutive-atlas/blob/main/docs/HISTORY_TRANSACTIONS.md',
                         'locator': 'Identity transport only; trusted complete legacy callback. ' + identity}],
            'fields': {field['slot']: {'shape': field['shape'], 'unit': copy.deepcopy(field['unit']),
                                      'storage': 'context' if field['transient'] or field['multiplier'] else 'state'}
                       for field in binding['layout']},
            'inputs': {}, 'phases': [{'name': 'transport', 'kind': 'advance',
                                     'base': 'parent', 'bindings': {}, 'writes': {}}]}


def _encode(binding, state):
    names = {field['name'] for field in binding['layout']}
    if not isinstance(state, dict) or set(state) != names:
        raise ValidationError('complete exact named legacy State required')
    return {field['slot']: copy.deepcopy(state[field['name']]) for field in binding['layout']}


def _decode(binding, values):
    return {field['name']: copy.deepcopy(values[field['slot']]) for field in binding['layout']}


def make_legacy_snapshot(model, state, parameters, initial_strain, *, generation=0):
    """Wrap complete supplied transport data; initial_strain is mandatory.

    No defaults, material initialization, admissibility checks, or physical-role
    inference occur. Parameters must be explicitly complete and within bounds.
    """
    _model(model)
    _six(initial_strain, 'legacy initial strain')
    binding = _binding(model, parameters)
    history = make_snapshot(_program(binding), _encode(binding, state), generation=generation,
                            preparation_epoch=0)
    strain = copy.deepcopy(initial_strain)
    return {'version': VERSION, 'binding': binding, 'history': history,
            'strain': strain, 'strain_sha256': _hash(strain)}


def _restore(model, parent):
    _model(model)
    if not isinstance(parent, dict) or set(parent) != _OUTER_KEYS or parent['version'] != VERSION:
        raise ValidationError('invalid complete legacy history snapshot')
    binding = parent['binding']
    if not isinstance(binding, dict) or set(binding) != _BINDING_KEYS:
        raise ValidationError('invalid legacy history binding')
    expected = _binding(model, binding['parameters'])
    if _hash(binding) != _hash(expected):
        raise ValidationError('legacy model/parameters/layout identity mismatch')
    _six(parent['strain'], 'legacy committed strain')
    if parent['strain_sha256'] != _hash(parent['strain']):
        raise ValidationError('legacy committed strain identity mismatch')
    history = parent['history']
    if not isinstance(history, dict) or set(history) != {'version', 'program_sha256', 'generation', 'preparation_epoch', 'values'}:
        raise ValidationError('invalid inner legacy history snapshot')
    if type(history['preparation_epoch']) is not int or history['preparation_epoch'] != 0:
        raise ValidationError('legacy transport does not support preparation epochs')
    program = _program(expected)
    rebuilt = make_snapshot(program, history['values'], generation=history['generation'], preparation_epoch=0)
    if history != rebuilt:
        raise ValidationError('legacy transport program identity mismatch')
    return expected, program, _decode(expected, rebuilt['values'])


def _diagnostics(value):
    if not isinstance(value, dict):
        raise ValidationError('legacy callback diagnostics must be a finite JSON object')
    stack = [(value, 0)]; nodes = 0
    while stack:
        item, depth = stack.pop(); nodes += 1
        if nodes > 10000 or depth > 64:
            raise ValidationError('legacy diagnostics exceed bounded JSON depth/size')
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValidationError('legacy diagnostic keys must be strings')
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif item is None or type(item) in (bool, str):
            pass
        elif not _number(item):
            raise ValidationError('legacy callback diagnostics contain nonfinite/non-JSON value')
    return copy.deepcopy(value)


def advance_legacy(model, parent, increment, callback):
    """One complete trusted callback, from copies of the ORIGINAL committed state.

    Callback signature: (model, named_state, six_increment, named_parameters).
    Return requires real Boolean ok, exact named state and finite JSON diagnostics.
    No callback events are published. Malformed/trial failures retain parent identity.
    """
    binding, program, state = _restore(model, parent)
    _six(increment, 'legacy caller increment')
    if not callable(callback):
        raise ValidationError('legacy complete-integrator callback must be callable')
    result = {'ok': False, 'snapshot': parent, 'events': [], 'diagnostics': {}}
    try:
        if parent['history']['generation'] >= MAX_COUNTER:
            raise ValidationError('legacy history generation limit exceeded')
        # Every argument is isolated, including the model itself and parameter data.
        response = callback(copy.deepcopy(model), copy.deepcopy(state), copy.deepcopy(increment),
                            copy.deepcopy(binding['parameters']))
        if not isinstance(response, dict) or type(response.get('ok')) is not bool:
            raise ValidationError('legacy callback requires exact Boolean ok')
        if 'diagnostics' not in response or 'state' not in response:
            raise ValidationError('legacy callback requires state and diagnostics')
        if 'message' in response and not isinstance(response['message'], str):
            raise ValidationError('legacy callback message must be a string')
        diagnostics = _diagnostics(response['diagnostics'])
        # Both success and reported failure must carry well-typed complete transport.
        encoded = _encode(binding, response['state'])
        history = make_snapshot(program, encoded, generation=parent['history']['generation'] + 1,
                                preparation_epoch=0)
        result['diagnostics'] = diagnostics
        if not response['ok']:
            result['message'] = response.get('message', 'legacy complete integration failed')
            return result
        strain = [a + b for a, b in zip(parent['strain'], increment)]
        _six(strain, 'legacy candidate strain')
        snapshot = {'version': VERSION, 'binding': copy.deepcopy(binding), 'history': history,
                    'strain': strain, 'strain_sha256': _hash(strain)}
        result.update(ok=True, snapshot=snapshot)
    except Exception as exc:
        # Trusted callback may raise arbitrary ordinary backend exceptions. Never
        # catch BaseException (KeyboardInterrupt/SystemExit); no parent was mutated.
        result['message'] = 'legacy history transaction failed; exact parent retained: ' + str(exc)
    return result
