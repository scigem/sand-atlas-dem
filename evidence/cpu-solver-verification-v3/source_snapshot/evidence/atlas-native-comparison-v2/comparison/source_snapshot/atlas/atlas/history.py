"""Experimental Python reference for typed immutable history transactions.

Not a material integrator, host adapter, or extension of model schema 0.1/0.2.
Every phase has an explicit base and simultaneous writes. Checks read the output
fields; write/guard expressions read only declared bindings. Failed trials return
the identical original snapshot object and no durable events.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
from pathlib import Path

from .dsl import has_value_only_dependency, FUNCTIONS, I, Sym, Type, ValidationError, dimensions, evaluate, infer, parse
from .structure import check

PROGRAM_VERSION = 'experimental-history-0.1'
SNAPSHOT_VERSION = 'experimental-history-snapshot-0.1'
MAX_COUNTER = 9007199254740991
_SCHEMA_ROOT = Path(__file__).resolve().parents[1] / 'schemas'
_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,63}\Z')
_RESERVED = set(FUNCTIONS) | {'I', 'select', 'parent'}


def _schema(name):
    return json.loads((_SCHEMA_ROOT / name).read_text())


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value) or value in _RESERVED:
        raise ValidationError('invalid or reserved history name')


def _type(field):
    return Type(field['shape'], dimensions(field['unit']))


def _hash(program):
    try:
        encoded = json.dumps(program, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError('history program is not finite JSON data') from exc
    return hashlib.sha256(encoded.encode()).hexdigest()


def _number(value):
    try:
        return (type(value) in (int, float) and math.isfinite(value)
                and (type(value) is float or float(value) == value))
    except OverflowError:
        return False


def _values(values, declarations, label):
    if not isinstance(values, dict) or set(values) != set(declarations):
        raise ValidationError(label + ': exact declared field set required')
    for name, field in declarations.items():
        value = values[name]
        if field['shape'] == 'sym':
            if not isinstance(value, list) or len(value) != 6 or not all(_number(x) for x in value):
                raise ValidationError(label + ': six finite Mandel components required: ' + name)
        elif not _number(value):
            raise ValidationError(label + ': finite scalar required: ' + name)
        if 'enum' in field and value not in field['enum']:
            raise ValidationError(label + ': invalid mode value: ' + name)


def _tree(expression, symbols, expected):
    try:
        tree = parse(expression)
        # Bound recursive inference/evaluation independently of Python stack size.
        stack = [(tree, 1)]
        while stack:
            node, depth = stack.pop()
            if depth > 128:
                raise ValidationError('history expression nesting exceeds 128')
            stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        if infer(tree, symbols) != expected:
            raise ValidationError('history expression shape or dimensions mismatch')
        # The legacy DSL infers boolean unary/additive expressions as boolean,
        # while Python evaluates them as numbers. Keep this new contract strict:
        # booleans may control select or be checked, never enter arithmetic.
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and not _number(node.value):
                raise ValidationError('history literals must be exactly representable in binary64')
            operands = ([node.operand] if isinstance(node, ast.UnaryOp) else
                        [node.left, node.right] if isinstance(node, ast.BinOp) else [])
            if any(infer(operand, symbols).shape == 'bool' for operand in operands):
                raise ValidationError('history arithmetic requires numeric operands')
        return tree
    except (ArithmeticError, RecursionError, TypeError) as exc:
        raise ValidationError('invalid history expression') from exc


def _compile_impl(program):
    check(program, _schema('history-program.schema.json'))
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', program['id']):
        raise ValidationError('invalid history program identity')
    if len(program['fields']) > 32 or len(program['inputs']) > 64:
        raise ValidationError('history declaration count exceeds limit')
    if any(not source['locator'].strip() for source in program['sources']):
        raise ValidationError('history source locator must be nonempty')
    fields, inputs = program['fields'], program['inputs']
    for declarations in (fields, inputs):
        for name, field in declarations.items():
            _name(name)
            _type(field)
    if sum(6 if field['shape'] == 'sym' else 1 for field in fields.values()) > 32:
        raise ValidationError('at most 32 scalarized retained history fields are supported')
    for field in fields.values():
        if field['storage'] == 'mode':
            if _type(field) != Type() or 'enum' not in field:
                raise ValidationError('history modes require dimensionless scalar enum')
            if not all(_number(x) for x in field['enum']) or len(set(field['enum'])) != len(field['enum']):
                raise ValidationError('mode enum must contain distinct finite numbers')
        elif 'enum' in field:
            raise ValidationError('enum is supported only for mode storage')
    available = {'parent': False}
    prepared = False
    compiled = []
    has_advance = False
    for index, phase in enumerate(program['phases']):
        name = phase['name']; _name(name)
        if name in available:
            raise ValidationError('duplicate history phase name')
        if len(phase['bindings']) > 64 or len(phase['writes']) > 32:
            raise ValidationError('history phase declaration count exceeds limit')
        prepare = phase['kind'] == 'prepare'
        if prepare and (index != 0 or phase['base'] != 'parent'):
            raise ValidationError('prepare is optional first phase with parent base only')
        has_advance |= not prepare
        base = phase['base']
        if base not in available:
            raise ValidationError('phase base must be parent or an earlier snapshot')
        if prepared and not available[base]:
            raise ValidationError('phase base must descend from prepared snapshot')
        symbols = {'I': Type('sym')}
        for alias, binding in phase['bindings'].items():
            _name(alias)
            if 'input' in binding:
                if binding['input'] not in inputs:
                    raise ValidationError('unknown history input binding')
                symbols[alias] = _type(inputs[binding['input']])
            else:
                snapshot = binding['snapshot']; field = binding['field']
                if snapshot not in available or field not in fields:
                    raise ValidationError('binding must read an earlier declared snapshot field')
                if prepared and not available[snapshot]:
                    raise ValidationError('binding cannot read pre-reset parent')
                symbols[alias] = _type(fields[field])
        guard = _tree(phase['guard'], symbols, Type()) if 'guard' in phase else None
        if guard is not None and has_value_only_dependency(guard):
            raise ValidationError('history event guards cannot depend on value-only spectral queries')
        writes = {}
        for target, expression in phase['writes'].items():
            if target not in fields:
                raise ValidationError('unknown history write target')
            if fields[target]['storage'] == 'preparation' and not prepare:
                raise ValidationError('preparation fields writable only in prepare phase')
            writes[target] = _tree(expression, symbols, _type(fields[target]))
        check_symbols = {'I': Type('sym')} | {key: _type(value) for key, value in fields.items()}
        checks = [_tree(expression, check_symbols, Type('bool')) for expression in phase.get('checks', [])]
        compiled.append((phase, guard, writes, checks))
        available[name] = prepare or available[base]
        prepared |= prepare
    if not has_advance:
        raise ValidationError('history advance program requires at least one advance phase')
    # Serialization also validates program-level JSON values prior to execution.
    _hash(program)
    return compiled


def _compile(program):
    try:
        return _compile_impl(program)
    except (ArithmeticError, TypeError, RecursionError) as exc:
        raise ValidationError('invalid history program data') from exc


def validate_program(program):
    """Validate structure, units, bindings and immutable phase ancestry; return program."""
    _compile(program)
    return program


def _counter(value):
    return type(value) is int and 0 <= value <= MAX_COUNTER


def make_snapshot(program, values, generation=0, preparation_epoch=0):
    """Construct a complete snapshot; no initialization/reset/equations are inferred."""
    validate_program(program)
    _values(values, program['fields'], 'history snapshot')
    if not _counter(generation) or not _counter(preparation_epoch):
        raise ValidationError('invalid history snapshot generation or preparation epoch')
    return {'version': SNAPSHOT_VERSION, 'program_sha256': _hash(program),
            'generation': generation, 'preparation_epoch': preparation_epoch,
            'values': copy.deepcopy(values)}


def _parent(program, parent):
    try:
        check(parent, _schema('history-snapshot.schema.json'))
    except (ArithmeticError, TypeError, RecursionError) as exc:
        raise ValidationError('invalid history snapshot data') from exc
    if parent['program_sha256'] != _hash(program):
        raise ValidationError('history snapshot program hash mismatch')
    if not _counter(parent['generation']) or not _counter(parent['preparation_epoch']):
        raise ValidationError('invalid history snapshot counters')
    _values(parent['values'], program['fields'], 'history parent')


def _environment(values):
    return {'I': I} | {key: Sym(tuple(value)) if isinstance(value, list) else value
                      for key, value in values.items()}


def _finite_result(value):
    if isinstance(value, Sym):
        return len(value.v) == 6 and all(_number(x) for x in value.v)
    return type(value) is bool or _number(value)


def _evaluate(tree, environment):
    """Use the shared DSL, checking only executed branches for finite arithmetic.

    Subexpression preflight ensures e.g. exp(1000)>0 cannot masquerade as a valid
    boolean check. select retains the DSL's lazy branch semantics. Evaluation is
    pure; repeated evaluation during preflight cannot advance material history.
    """
    if isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name) and tree.func.id == 'select':
        condition = _evaluate(tree.args[0], environment)
        result = _evaluate(tree.args[1] if condition else tree.args[2], environment)
    else:
        children = tree.args if isinstance(tree, ast.Call) else ast.iter_child_nodes(tree)
        for child in children:
            if isinstance(child, ast.expr):
                _evaluate(child, environment)
        result = evaluate(tree, environment)
    if not _finite_result(result):
        raise ValidationError('nonfinite history expression result')
    return result


def advance(program, parent, inputs):
    """Evaluate a pure caller transaction; malformed inputs raise ValidationError.

    Arithmetic, output-domain/mode and phase-check failures return ok=False,
    snapshot=parent (object identity preserved), events=[], attempted diagnostics.
    No internal integration/subdivision, PM4 equations or native adapter is implied.
    """
    compiled = _compile(program)
    _parent(program, parent)
    _values(inputs, program['inputs'], 'history inputs')
    diagnostics = {'attempted_phases': 0, 'executed_phases': 0, 'failed_phase': None}
    result = {'ok': False, 'snapshot': parent, 'events': [], 'diagnostics': diagnostics}
    snapshots = {'parent': copy.deepcopy(parent['values'])}
    events = []; prepare_ran = False
    try:
        for phase, guard, writes, checks in compiled:
            diagnostics['attempted_phases'] += 1
            diagnostics['failed_phase'] = phase['name']
            bound = {}
            for alias, binding in phase['bindings'].items():
                bound[alias] = inputs[binding['input']] if 'input' in binding else snapshots[binding['snapshot']][binding['field']]
            environment = _environment(bound)
            fires = guard is None or _evaluate(guard, environment) < 0
            output = copy.deepcopy(snapshots[phase['base']])
            if fires:
                diagnostics['executed_phases'] += 1
                for target, expression in writes.items():
                    value = _evaluate(expression, environment)
                    output[target] = list(value.v) if isinstance(value, Sym) else value
                _values(output, program['fields'], 'history trial')
                prepare_ran |= phase['kind'] == 'prepare'
            check_environment = _environment(output)
            if any(_evaluate(expression, check_environment) is not True for expression in checks):
                raise ValidationError('history output check failed')
            snapshots[phase['name']] = output
            if fires:
                events.append({'phase': phase['name'], 'kind': phase['kind']})
        generation = parent['generation'] + 1
        epoch = parent['preparation_epoch'] + int(prepare_ran)
        if not _counter(generation) or not _counter(epoch):
            raise ValidationError('history snapshot counter limit exceeded')
        snapshot = {'version': SNAPSHOT_VERSION, 'program_sha256': parent['program_sha256'],
                    'generation': generation, 'preparation_epoch': epoch,
                    'values': snapshots[program['phases'][-1]['name']]}
        diagnostics['failed_phase'] = None
        result.update(ok=True, snapshot=snapshot, events=events)
    except (ArithmeticError, ValueError, TypeError, RecursionError) as exc:
        result['message'] = 'history transaction failed; exact parent retained: ' + str(exc)
    return result
