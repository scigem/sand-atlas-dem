"""Dimension-checked initialization recipes, separate from material integration.

Experimental Python preprocessing: no model-specific mathematical kernel and no
inferred preparation. Outputs are explicit fields for the common timed driver.
"""
import hashlib
import json
import math
from .dsl import I, Sym, Type, ValidationError, evaluate, infer, parse
from .model import ROOT, IDENT, RESERVED, evaluation_environment, field_type, observe, physical_time, state_role, validate
from .structure import check


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def initialize_state(model, state, parameters, *, loading=None, time=None):
    """Validate supplied persistent state under the v0.3 initialization contract.

    Never substitute example values, project state, or infer a preparation history.
    Temporary solve fields are zero-filled for the complete engine layout.
    """
    validate(model)
    if model['schema_version'] != '0.3':
        raise ValidationError('explicit state initialization requires model v0.3')
    required = model['initialization']['required_state']
    if not isinstance(state, dict) or set(state) != set(required):
        raise ValidationError('all and only persistent state fields are required')
    if not isinstance(parameters, dict) or set(parameters) != set(model['parameters']):
        raise ValidationError('complete named initialization parameters are required')
    for name, field in model['parameters'].items():
        value = parameters[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not field['min'] <= value <= field['max']:
            raise ValidationError(f'{name}: initialization parameter must be finite and within bounds')
    physical = physical_time(model)
    if physical:
        if type(time) not in (int, float) or not math.isfinite(time) or loading is None:
            raise ValidationError('physical-time initialization requires explicit time and strain rate')
    else:
        if time is not None or loading is not None:
            raise ValidationError('incremental initialization uses zero strain increment and no clock')
        loading = [0.0]*6
    if not isinstance(loading, (list, tuple)) or len(loading) != 6 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in loading):
        raise ValidationError('initial loading requires six finite Mandel components')
    named = {}
    for field in model['state']:
        name = field['name']
        value = state[name] if state_role(field) == 'persistent' else ([0.0]*6 if field['shape'] == 'sym' else 0.0)
        if field['shape'] == 'sym':
            if not isinstance(value, (list, tuple)) or len(value) != 6:
                raise ValidationError(f'{name}: six Mandel components required')
            components = list(value)
        else:
            components = [value]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in components):
            raise ValidationError(f'{name}: finite state values required')
        named[name] = components if field['shape'] == 'sym' else value
    try:
        for field in model['state']:
            scale = evaluate(parse(field['scale']), parameters)
            if not math.isfinite(scale) or scale <= 0:
                raise ValidationError('initialization state scale must be positive at supplied parameters')
        env = evaluation_environment(model, named, None if physical else named,
                                     loading, parameters, time=time)
        if not all(evaluate(parse(expr), env) for expr in model.get('initial_checks', [])):
            raise ValidationError('initial state fails declared initial_checks')
        observations = observe(model, named, loading, parameters, time=time) if physical else {}
        observations = {k: list(v.v) if isinstance(v, Sym) else v for k, v in observations.items()}
        if any(not math.isfinite(x) for v in observations.values() for x in (v if isinstance(v, list) else [v])):
            raise ValidationError('nonfinite initial observable')
    except (ArithmeticError, ValueError) as exc:
        raise ValidationError(f'initialization rejected: {exc}') from exc
    flat = [value for field in model['state'] for value in
            (named[field['name']] if field['shape'] == 'sym' else [named[field['name']]])]
    return {'version': 'experimental-explicit-state-initialization-0.1',
            'named_state': named, 'initial': flat, 'parameters': dict(parameters),
            'initial_time': time, 'initial_loading': list(loading), 'observables': observations,
            'provenance': {'model_sha256': canonical_hash(model),
                           'validation': 'declared_domain_and_initial_checks_only',
                           'preparation_history': 'not_inferred'}}


def validate_recipe(recipe, model):
    check(recipe,json.loads((ROOT/'schemas/initialization-recipe.schema.json').read_text()))
    validate(model)
    if not physical_time(model) or recipe['model'] != model['id']:
        raise ValidationError('initialization recipe requires its matching physical-time model')
    if recipe['model_sha256'] != canonical_hash(model):
        raise ValidationError('initialization recipe model hash mismatch; reconcile model revision')
    if not recipe['id'].strip() or any(not s[k].strip() for s in recipe['sources'] for k in ('url','locator')):
        raise ValidationError('recipe identity and source locators must be nonempty')
    symbols={'I':Type('sym'), **{name:field_type(f) for name,f in model['parameters'].items()}}
    def declare(name,field):
        if not IDENT.fullmatch(name) or name in RESERVED or name in symbols or name.endswith('_n'):
            raise ValidationError(f'invalid or colliding initialization name: {name}')
        symbols[name]=field_type(field)
    for name,field in recipe['inputs'].items():declare(name,field)
    for term in recipe['terms']:
        if not term['locator'].strip():raise ValidationError('term source locator required')
        for guard in term.get('requires',[]):
            if infer(parse(guard),symbols)!=Type('bool'):
                raise ValidationError('initialization guards must be comparisons')
        if infer(parse(term['expr']),symbols)!=field_type(term):
            raise ValidationError(f"{term['name']}: initialization term dimensions or shape mismatch")
        declare(term['name'],term)
    if set(recipe['state']) != {f['name'] for f in model['state']}:
        raise ValidationError('recipe must produce all and only model state fields')
    for field in model['state']:
        if infer(parse(recipe['state'][field['name']]),symbols)!=field_type(field):
            raise ValidationError('initialization state shape or dimensions mismatch')
    if infer(parse(recipe['initial_strain_rate']),symbols)!=Type('sym',(0,0,-1)):
        raise ValidationError('initial strain rate must be a symmetric tensor per second')
    return recipe


def initialize(recipe, configuration, model):
    """Resolve a recipe, check model domain, and return states plus provenance.

    All parameters/inputs are mandatory and named; model defaults never fill gaps.
    Guarded ordered terms are pure expressions, not nonlinear solves or projection.
    """
    validate_recipe(recipe,model)
    check(configuration,json.loads((ROOT/'schemas/initialization-input.schema.json').read_text()))
    parameters=configuration['parameters']; inputs=configuration['inputs']
    if set(parameters)!=set(model['parameters']) or set(inputs)!=set(recipe['inputs']):
        raise ValidationError('all and only declared initialization inputs and model parameters required')
    for name,field in model['parameters'].items():
        if not field['min']<=parameters[name]<=field['max']:
            raise ValidationError(f'{name}: parameter outside model bounds')
    env={'I':I,**parameters}
    for name,field in recipe['inputs'].items():
        value=inputs[name]; symmetric=field.get('shape')=='sym'
        if symmetric!=isinstance(value,list):raise ValidationError(f'{name}: initialization input shape mismatch')
        env[name]=Sym(tuple(value)) if symmetric else value
    def finite(value):
        if isinstance(value,Sym):values=value.v
        else:values=[value]
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
            raise ValidationError('nonfinite initialization result')
        return value
    try:
        for term in recipe['terms']:
            if not all(evaluate(parse(g),env) for g in term.get('requires',[])):
                raise ValidationError(f"{term['name']}: initialization guard failed")
            env[term['name']]=finite(evaluate(parse(term['expr']),env))
        state={name:finite(evaluate(parse(expr),env)) for name,expr in recipe['state'].items()}
        rate=finite(evaluate(parse(recipe['initial_strain_rate']),env))
        unbox=lambda v:list(v.v) if isinstance(v,Sym) else v
        named_state={k:unbox(v) for k,v in state.items()}
        if model['schema_version'] == '0.3':
            observations=initialize_state(model,named_state,parameters,loading=rate.v,time=0)['observables']
        else:
            initial_env=evaluation_environment(model,named_state,None,rate.v,parameters,time=0)
            if not all(evaluate(parse(g),initial_env) for g in model.get('initial_checks',[])):
                raise ValidationError('model initial-history checks failed')
            observations=observe(model,named_state,rate.v,parameters,time=0)
            observations={k:unbox(finite(v)) for k,v in observations.items()}
    except (ArithmeticError,ValueError) as exc:
        raise ValidationError(f'initialization rejected: {exc}') from exc
    flat=[]
    for field in model['state']:
        value=named_state[field['name']];flat.extend(value if isinstance(value,list) else [value])
    fields={'model':model['id'],'parameters':[parameters[k] for k in model['parameters']],
            'initial':flat,'initial_strain_rate':list(rate.v)}
    return {'version':'experimental-initialization-result-0.1','run_fields':fields,
            'named_state':named_state,'observables_at_time_zero':observations,
            'provenance':{'recipe_id':recipe['id'],'recipe_sha256':canonical_hash(recipe),
                          'configuration_sha256':canonical_hash(configuration),
                          'model_sha256':canonical_hash(model),'run_fields_sha256':canonical_hash(fields),
                          'sources':recipe['sources'],
                          'limitations':['Algebraic initial state, not a preparation history or calibration.',
                                         'Experimental Python preprocessing; not a browser/native recipe API.']}}
