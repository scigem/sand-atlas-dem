"""Model loading and semantic validation. JSON is the canonical YAML subset."""
from __future__ import annotations
import ast
import json
import math
import re
from pathlib import Path
from .dsl import has_value_only_dependency, DIMS, TIME, I, Sym, Type, ZERO, dimensions, ValidationError, cpp, evaluate, infer, parse

MAX_STATE_COMPONENTS = 32
ROOT = Path(__file__).resolve().parents[1]
IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
RESERVED = {"I", "de", "dt", "time", "time_n", "strain_rate", "loading", "evaluation_time", "check_only", "observables_only", "check_initial", "T", "params", "z", "zn", "out", "domain", "true", "false",
            "double", "auto", "if", "else", "return", "const", "class", "struct", "template",
            "int", "float", "void", "bool", "new", "delete", "switch", "case", "default",
            "for", "while", "do", "break", "continue", "throw", "try", "catch", "namespace",
            "make_sym", "axisymmetric_x", "atlas_eval_ok", "using", "typename", "static", "virtual", "public", "private", "protected"}


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    model = json.loads(Path(path).read_text(), object_pairs_hook=reject_duplicate_keys,
                       parse_constant=lambda s: (_ for _ in ()).throw(ValidationError(f"invalid number: {s}")))
    validate(model)
    return model


def field_type(field):
    if field.get('shape', 'scalar') not in ('scalar', 'sym'):
        raise ValidationError('unsupported shape')
    return Type(field.get('shape','scalar'), dimensions(field.get('unit','1')))


def validate_scale_operators(node):
    if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('expm','log1p') for n in ast.walk(node)):
        raise ValidationError('checked expm/log1p are unsupported in state/observable scale expressions; use an explicit positive reference scale')


def physical_time(model):
    return (model.get('formulation', {}).get('time_dependence') == 'physical_time'
            if model.get('schema_version') == '0.3' else model.get('schema_version') == '0.2')


def integration_kind(model):
    """Engine dispatch; equation meaning is explicit in v0.3 documents."""
    if model.get('schema_version') == '0.3':
        return {'incremental_residual': 'residual', 'strain_path_rate': 'rate',
                'ode': 'rate'}[model['formulation']['equation_form']]
    return 'rate' if physical_time(model) else model['integration']


def formulation_inputs(*, physical):
    """Canonical input descriptors for the currently executable profiles.

    Equation symbols and update-call arguments are separate namespaces. Every
    listed input is required, even when a particular equation does not use it.
    """
    if not physical:
        return {'equation': {'de': {
            'quantity': 'strain_increment', 'measure': 'infinitesimal_strain',
            'shape': 'sym', 'unit': '1', 'configuration': 'current',
            'sampling': 'increment', 'constraint': 'finite',
        }}, 'advance': {}}
    return {'equation': {
        'time': {'quantity': 'time', 'shape': 'scalar', 'unit': 's',
                 'sampling': 'stage', 'constraint': 'finite'},
        'strain_rate': {'quantity': 'strain_rate', 'measure': 'infinitesimal_strain_rate',
                        'shape': 'sym', 'unit': {'si': [0, 0, -1, 0]},
                        'configuration': 'current', 'sampling': 'stage', 'constraint': 'finite'},
    }, 'advance': {
        'start_time': {'quantity': 'time', 'shape': 'scalar', 'unit': 's',
                       'sampling': 'start', 'constraint': 'finite'},
        'duration': {'quantity': 'duration', 'shape': 'scalar', 'unit': 's',
                     'sampling': 'interval', 'constraint': 'positive_finite'},
    }}


def equation_input_types(model):
    """Expression input types from a validated contract (legacy profiles included)."""
    inputs = (model['formulation']['inputs'] if model.get('schema_version') == '0.3'
              else formulation_inputs(physical=physical_time(model)))
    return {name: field_type(field) for name, field in inputs['equation'].items()}


def state_role(field):
    """Storage/solve role, with legacy flag support for preserved definitions."""
    if 'semantics' in field:
        return field['semantics']['role']
    return ('plastic_multiplier' if field.get('multiplier') else
            'increment_unknown' if field.get('transient') else 'persistent')


def migrate_formulation(model, *, stress_scope=None):
    """Return a validated v0.3 copy; never rewrite a hash-pinned definition.

    Physical-time stress scope must be supplied by the caller from source evidence.
    """
    import copy
    validate(model)
    result = copy.deepcopy(model)
    if model['schema_version'] == '0.3':
        if stress_scope is not None and model['formulation']['stress'].get('scope') != stress_scope:
            raise ValidationError('stress scope conflicts with existing formulation')
        return result
    physical = physical_time(model)
    if physical:
        has_stress = any(f['name'] == 'stress' for f in model.get('observables', []))
        if has_stress and stress_scope not in ('effective', 'total'):
            raise ValidationError('physical-time stress migration requires explicit effective/total scope')
        if not has_stress and stress_scope is not None:
            raise ValidationError('cannot assign stress scope without a stress observable')
        stress = ({'source': 'observable', 'name': 'stress', 'measure': 'cauchy', 'scope': stress_scope}
                  if has_stress else {'source': 'none'})
    else:
        if stress_scope not in (None, 'effective'):
            raise ValidationError('v0.1 stress is effective stress')
        stress = {'source': 'state', 'name': 'sigma', 'measure': 'cauchy', 'scope': 'effective'}
    result['formulation'] = {
        'time_dependence': 'physical_time' if physical else 'strain_path',
        'equation_form': 'ode' if physical else {'residual': 'incremental_residual', 'rate': 'strain_path_rate'}[model['integration']],
        'kinematics': model.get('formulation', {}).get('kinematics', 'small_strain_3d'),
        'frame': 'fixed_spatial', 'objective_rate': 'none', 'tensor_representation': 'mandel',
        'stress_sign': 'compression_positive', 'strain_sign': 'compression_positive', 'stress': stress,
        'inputs': formulation_inputs(physical=physical),
    }
    result.pop('integration', None)
    for field in result['state']:
        role = state_role(field)
        is_stress = not physical and field['name'] == 'sigma'
        field['semantics'] = {
            'role': role,
            'quantity': 'cauchy_stress' if is_stress else 'unspecified',
            'transformation': 'symmetric_spatial_tensor' if is_stress else 'fixed_components',
        }
        field.pop('transient', None)
        field.pop('multiplier', None)
    result['initialization'] = {
        'method': 'explicit_state', 'default_policy': 'examples_only',
        'required_state': [f['name'] for f in result['state'] if state_role(f) == 'persistent'],
        'validation': 'domain_and_initial_checks',
    }
    result['schema_version'] = '0.3'
    validate(result)
    return result


def validate(model):
    from .structure import check
    version=model.get('schema_version')
    if version not in ('0.1','0.2','0.3'):
        raise ValidationError('unsupported schema version')
    schema=f'model-v{version}.schema.json' if version != '0.1' else 'model.schema.json'
    check(model,json.loads((ROOT/'schemas'/schema).read_text()))
    physical=physical_time(model)
    required = {'schema_version','id','name','family','status','sources','parameters','state', 'derived','equations','domain','description'}
    if not required <= model.keys():
        raise ValidationError(f"missing fields: {sorted(required-model.keys())}")
    if not physical and integration_kind(model) not in ('residual','rate'):
        raise ValidationError('unsupported schema version or integration kind')
    if not re.fullmatch(r'[a-z][a-z0-9-]*',model['id']):
        raise ValidationError('invalid model ID')
    if model['status'] not in ('draft','transcribed','numerically-tested','reviewed'):
        raise ValidationError('invalid review status')
    if not model['sources'] or any(not {'url','locator','role'} <= s.keys() for s in model['sources']):
        raise ValidationError('source URL, locator and role are required')
    symbols = {'I':Type('sym'), **equation_input_types(model)}
    advance_inputs = set(model['formulation']['inputs']['advance']) if version == '0.3' else set()
    names = set()
    def declare(name, typ):
        if not IDENT.fullmatch(name) or name in RESERVED or name in advance_inputs or name.endswith('_n') or name in names:
            raise ValidationError(f'invalid, duplicate or reserved identifier: {name}')
        names.add(name)
        symbols[name]=typ
    for name,field in model['parameters'].items():
        typ=field_type(field)
        if typ.shape != 'scalar': raise ValidationError('parameters must be scalar')
        declare(name,typ)
        if any(type(field.get(k)) not in (int,float) or not math.isfinite(field[k]) for k in ('default','min','max')):
            raise ValidationError(f'{name}: finite default, min and max required')
        if not field['min'] <= field['default'] <= field['max']:
            raise ValidationError(f'{name}: default outside bounds')
    if not physical and (not model['state'] or model['state'][0]['name'] != 'sigma' or field_type(model['state'][0]) != Type('sym',DIMS['Pa'])):
        raise ValidationError('first state must be symmetric stress sigma in Pa')
    count=0
    for field in model['state']:
        typ=field_type(field); name=field['name']
        declare(name,typ)
        role = state_role(field)
        if not physical and (version != '0.3' or role == 'persistent'): symbols[name+'_n']=typ
        initial=field['initial']
        if typ.shape=='sym':
            if not isinstance(initial,list) or len(initial)!=6: raise ValidationError(f'{name}: six Mandel components required')
        else: initial=[initial]
        if any(type(v) not in (float,int) or not math.isfinite(v) for v in initial): raise ValidationError('nonfinite initial value')
        scale=parse(field['scale'])
        validate_scale_operators(scale)
        param_symbols={key:field_type(value) for key,value in model['parameters'].items()}
        if infer(scale,param_symbols)!=Type(dim=typ.dim): raise ValidationError(f'{name}: scale must have matching units and reference parameters only')
        env={k:v['default'] for k,v in model['parameters'].items()}
        scale_value=evaluate(scale,env)
        if not math.isfinite(scale_value) or scale_value<=0: raise ValidationError('state scale must be finite and positive')
        count+=6 if typ.shape=='sym' else 1
        if physical and role != 'persistent':
            raise ValidationError('physical-time ODE state must be persistent; incremental multipliers are unsupported')
        if role == 'plastic_multiplier' and (typ!=Type() or (version != '0.3' and not field.get('transient')) or integration_kind(model)!='residual'):
            raise ValidationError('multipliers must be dimensionless transient scalars in residual models')
        if version == '0.3':
            transformation = field['semantics']['transformation']
            if ((transformation == 'scalar_invariant' and typ.shape != 'scalar') or
                    (transformation == 'symmetric_spatial_tensor' and typ.shape != 'sym')):
                raise ValidationError('state transformation must match declared shape')
            if not field['semantics']['quantity'].strip():
                raise ValidationError('state physical quantity must be nonempty; use unspecified for unresolved meaning')
            if role != 'persistent' and (integration_kind(model) != 'residual' or any(v != 0 for v in initial)):
                raise ValidationError('temporary solve state requires a residual profile and zero initial values')
    if count>MAX_STATE_COMPONENTS: raise ValidationError(f'at most {MAX_STATE_COMPONENTS} state components are supported')
    value_only_dependencies=set()
    for field in model['derived']:
        tree=parse(field['expr'])
        if has_value_only_dependency(tree,value_only_dependencies):value_only_dependencies.add(field['name'])
        typ=infer(tree,symbols)
        if typ.shape=='bool': raise ValidationError('derived values must be numeric')
        declare(field['name'],typ)
    reset_targets=set()
    state_fields={f['name']:f for f in model['state']}
    for update in model.get('increment_reset',{}).get('updates',[]):
        target=update['target'];field=state_fields.get(target)
        if field is None or target in reset_targets or state_role(field) != 'persistent':
            raise ValidationError('increment reset targets must be distinct persistent state fields')
        reset_targets.add(target)
        guard_tree=parse(update['guard'])
        guard_type=infer(guard_tree,symbols)
        if guard_type.shape!='scalar' or guard_type.value_only or has_value_only_dependency(guard_tree,value_only_dependencies):
            raise ValidationError('increment reset guard must be a signed scalar expression')
        if infer(parse(update['value']),symbols)!=field_type(field):
            raise ValidationError('increment reset value must match target shape and dimensions')
    eqs=model['equations']
    if set(eqs)!={f['name'] for f in model['state']}: raise ValidationError('one equation per state is required')
    for f in model['state']:
        expected=field_type(f)
        if physical: expected=Type(expected.shape,tuple(a-b for a,b in zip(expected.dim,TIME)))
        if infer(parse(eqs[f['name']]),symbols)!=expected: raise ValidationError(f"{f['name']}: equation shape or dimensions mismatch")
    for condition in model['domain']:
        if infer(parse(condition),symbols)!=Type('bool'): raise ValidationError('domain checks must be comparisons')
    for condition in model.get('initial_checks',[]):
        if infer(parse(condition),symbols)!=Type('bool'): raise ValidationError('initial checks must be comparisons')
    for field in model.get('observables',[]):
        typ=field_type(field)
        if infer(parse(field['expr']),symbols)!=typ:
            raise ValidationError(f"{field['name']}: observable shape or dimensions mismatch")
        if field['name']=='stress' and typ!=Type('sym',DIMS['Pa']):
            raise ValidationError('stress observable must be a symmetric tensor in Pa')
        scale=parse(field['scale'])
        validate_scale_operators(scale)
        param_symbols={name:field_type(value) for name,value in model['parameters'].items()}
        scale_value=evaluate(scale,defaults(model)) if infer(scale,param_symbols)==Type(dim=typ.dim) else float('nan')
        if not math.isfinite(scale_value) or scale_value<=0:
            raise ValidationError('observable scale must be positive, parameter-only, and have matching units')
        # Output names do not enter the equation/derived namespace.
        if not IDENT.fullmatch(field['name']) or field['name'] in RESERVED or field['name'].endswith('_n') or field['name'] in names:
            raise ValidationError('invalid or duplicate observable name')
        names.add(field['name'])
    if version == '0.3':
        persistent = [f['name'] for f in model['state'] if state_role(f) == 'persistent']
        if model['initialization']['required_state'] != persistent:
            raise ValidationError('initialization must require every persistent state field in layout order')
        stress = model['formulation']['stress']
        if not physical:
            if stress != {'source': 'state', 'name': 'sigma', 'measure': 'cauchy', 'scope': 'effective'}:
                raise ValidationError('incremental profile requires effective Cauchy stress state sigma')
            if state_role(model['state'][0]) != 'persistent':
                raise ValidationError('stress output sigma must be persistent')
            semantics = model['state'][0]['semantics']
            if semantics['quantity'] != 'cauchy_stress' or semantics['transformation'] != 'symmetric_spatial_tensor':
                raise ValidationError('stress sigma must declare Cauchy-stress spatial-tensor semantics')
        else:
            outputs = {f['name']: f for f in model.get('observables', [])}
            if stress['source'] == 'none':
                if 'stress' in outputs:
                    raise ValidationError('stress observable must be bound by the formulation')
            elif stress['name'] != 'stress' or stress['name'] not in outputs or field_type(outputs[stress['name']]) != Type('sym', DIMS['Pa']):
                raise ValidationError('physical-time stress must bind the symmetric Pa observable stress')
    return model


def defaults(model):
    return {name: p['default'] for name,p in model['parameters'].items()}


def axisymmetric_x(values):
    if len(values)!=6 or any(not math.isfinite(v) for v in values):return False
    tolerance=64*__import__('sys').float_info.epsilon*max(map(abs,values))
    return abs(values[1]-values[2])<=tolerance and all(abs(v)<=tolerance for v in values[3:])


def evaluation_environment(model, state, previous, increment, parameters=None, *, time=None, _check_domain=True):
    if model.get('schema_version') == '0.3' and (len(increment) != 6 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in increment)):
        raise ValueError('declared loading input requires six finite Mandel components')
    if physical_time(model) and (type(time) not in (int,float) or not math.isfinite(time)):
        raise ValueError('physical-time evaluation requires finite stage time')
    if model.get('formulation',{}).get('kinematics')=='axisymmetric_x' and not axisymmetric_x(increment):
        raise ValueError('axisymmetric_x requires equal radial rates and zero shear')
    env={'I':I,'de':Sym(tuple(increment)),**defaults(model),**(parameters or {})}
    if physical_time(model): env.update(time=time, strain_rate=Sym(tuple(increment)))
    for f in model['state']:
        name=f['name']; wrap=lambda v: Sym(tuple(v)) if f.get('shape')=='sym' else v
        env[name]=wrap(state[name])
        if not physical_time(model): env[name+'_n']=wrap(previous[name])
    for f in model['derived']:
        v=env[f['name']]=evaluate(parse(f['expr']),env)
        if any(not math.isfinite(x) for x in (v.v if isinstance(v,Sym) else [v])):
            raise ValueError('nonfinite derived value')
    if _check_domain and not all(evaluate(parse(c),env) for c in model['domain']): raise ValueError('state outside model domain')
    return env


def prepare_increment(model, state, increment, parameters=None):
    """Pure committed-parent reset, before any internal integration or trial stage."""
    if physical_time(model):raise ValidationError('parent-increment resets require a legacy incremental definition')
    if len(increment)!=6 or any(not math.isfinite(v) for v in increment):
        raise ValueError('six finite Mandel increment components required')
    values={**defaults(model),**(parameters or {})}
    if set(values)!=set(model['parameters']) or any(type(values[n]) not in (int,float) or not math.isfinite(values[n]) or not f['min']<=values[n]<=f['max'] for n,f in model['parameters'].items()):
        raise ValueError('invalid increment reset parameters')
    if set(state)!={f['name'] for f in model['state']}:
        raise ValueError('increment reset state layout mismatch')
    for f in model['state']:
        v=state[f['name']];components=v if f['shape']=='sym' else [v]
        if (f['shape']=='sym' and len(v)!=6) or any(type(x) not in (int,float) or not math.isfinite(x) for x in components):
            raise ValueError('invalid increment reset state')
        scale=evaluate(parse(f['scale']),values)
        if not math.isfinite(scale) or scale<=0:raise ValueError('invalid state scale')
    initial_env=evaluation_environment(model,state,state,[0]*6,values)
    if not all(evaluate(parse(c),initial_env) for c in model.get('initial_checks',[])):
        raise ValueError('inadmissible parent state')
    env=evaluation_environment(model,state,state,increment,values,_check_domain=False)
    prepared={f['name']:list(state[f['name']]) if f.get('shape')=='sym' else state[f['name']] for f in model['state']}
    for update in model.get('increment_reset',{}).get('updates',[]):
        guard=evaluate(parse(update['guard']),env)
        if not math.isfinite(guard):raise ValueError('nonfinite increment reset guard')
        if guard<0:
            v=evaluate(parse(update['value']),env)
            prepared[update['target']]=list(v.v) if isinstance(v,Sym) else v
    if any(not math.isfinite(v) for f in model['state'] for v in (prepared[f['name']] if f.get('shape')=='sym' else [prepared[f['name']]])):
        raise ValueError('nonfinite increment reset state')
    evaluation_environment(model,prepared,prepared,[0]*6,values)
    return prepared


def interpret(model, state, previous, increment, parameters=None, *, time=None):
    env=evaluation_environment(model,state,previous,increment,parameters,time=time)
    return {k:evaluate(parse(expr),env) for k,expr in model['equations'].items()}


def observe(model,state,strain_rate,parameters=None,*,time=None):
    if not physical_time(model):raise ValidationError('observables require a physical-time ODE definition')
    env=evaluation_environment(model,state,None,strain_rate,parameters,time=time)
    outputs={f['name']:evaluate(parse(f['expr']),env) for f in model.get('observables',[])}
    if model.get('formulation',{}).get('kinematics')=='axisymmetric_x' and 'stress' in outputs and not axisymmetric_x(outputs['stress'].v):
        raise ValueError('axisymmetric_x stress observable is outside its declared scope')
    return outputs


def all_models():
    return [load(p) for p in sorted((ROOT/'models').glob('*.json'))]
