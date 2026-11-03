"""Explicit piecewise-linear caller-path refinement, separate from material tolerances.

No fitting, sorting, deduplication, physical-time interpolation or extrapolation.
The finest recorded run is a numerical comparator, never an external reference.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from .compiler import generate
from .dsl import evaluate, parse, ValidationError
from .model import ROOT, all_models, state_role
from .recorded import execute_recorded, validate_recorded_run
from .structure import check

VERSION='experimental-recorded-refinement-0.1'


def refine_recorded(spec, factor):
    """Insert factor-1 explicit linear target knots in each original interval.

    Return a new specification plus indices of all original observations. Exact
    endpoint objects/values are preserved. Repeated targets stay repeated.
    """
    validate_recorded_run(spec)
    if type(factor) is not int or factor<1:raise ValidationError('positive integer refinement factor required')
    source=spec['path']['knots']
    if 1+(len(source)-1)*factor>5001:raise ValidationError('refined path exceeds 5001 knots')
    result=copy.deepcopy(spec);knots=[copy.deepcopy(source[0])];indices=[0]
    for left,right in zip(source,source[1:]):
        for j in range(1,factor+1):
            if j==factor:knots.append(copy.deepcopy(right))
            else:
                t=j/factor
                knots.append(dict(coordinate=(1-t)*left['coordinate']+t*right['coordinate'],
                                  targets=[(1-t)*a+t*b for a,b in zip(left['targets'],right['targets'])]))
        indices.append(len(knots)-1)
    result['path']['knots']=knots
    validate_recorded_run(result)  # Reject nonfinite or rounded/coincident inserted coordinates.
    return result,indices


def state_layout(model,parameters):
    if len(parameters)!=len(model['parameters']):raise ValueError('parameter layout mismatch')
    env=dict(zip(model['parameters'],parameters));fields=[];offset=0
    for field in model['state']:
        size=6 if field['shape']=='sym' else 1;scale=evaluate(parse(field['scale']),env)
        if not math.isfinite(scale) or scale<=0:raise ValueError('invalid comparison state scale')
        fields.append(dict(name=field['name'],unit=field['unit'],offset=offset,size=size,scale=scale,
                           transient=state_role(field) != 'persistent'))
        offset+=size
    return fields


def difference(a,b,fields):
    """Max errors at matching original observations, excluding local multipliers
    from the persistent-state norm while still reporting each transient field.
    """
    if len(a)!=len(b) or not a:raise ValueError('same nonempty original observation grid required')
    width=sum(f['size'] for f in fields)
    field_errors={f['name']:0. for f in fields};strain_error=0.;persistent=0.;transient=0.
    for x,y in zip(a,b):
        if x['coordinate']!=y['coordinate']:raise ValueError('original observation coordinates differ')
        if len(x['state'])!=width or len(y['state'])!=width or len(x['strain'])!=6 or len(y['strain'])!=6:raise ValueError('comparison state layout mismatch')
        values=x['state']+y['state']+x['strain']+y['strain']
        if any(not math.isfinite(v) for v in values):raise ValueError('nonfinite accepted comparison state')
        strain_error=max(strain_error,max(abs(v-w) for v,w in zip(x['strain'],y['strain'])))
        for f in fields:
            start=f['offset'];end=start+f['size'];delta=max(abs(x['state'][j]-y['state'][j]) for j in range(start,end))
            field_errors[f['name']]=max(field_errors[f['name']],delta)
            if f['transient']:transient=max(transient,delta/f['scale'])
            else:persistent=max(persistent,delta/f['scale'])
    return dict(max_strain_component_abs=strain_error,max_persistent_state_scaled=persistent,
                max_transient_state_scaled=transient,field_max_abs=field_errors)


def validate_study(config):
    check(config,json.loads((ROOT/'schemas/recorded-refinement.schema.json').read_text()))
    factors=config['factors']
    if factors[0]!=1 or any(b<=a or b%a for a,b in zip(factors,factors[1:])):
        raise ValidationError('refinement factors must start at 1 and form strictly increasing nested multiples')
    ids=[case['id'] for case in config['cases']]
    if len(ids)!=len(set(ids)):raise ValidationError('duplicate refinement case ID')
    for case in config['cases']:
        validate_recorded_run(case['specification'])
        # Bound work and verify the finest grid before executing any solver.
        refine_recorded(case['specification'],factors[-1])
    return config


def execute_study(config,solver,output):
    """Run serially, retain every independent level, then compare full successes.

    Never substitute the finest successful run when the requested finest fails.
    The output is numerical evidence; it has no scientific pass threshold.
    """
    validate_study(config);solver=Path(solver).resolve();output=Path(output)
    definitions={m['id']:m for m in all_models()}
    for case in config['cases']:
        if case['specification']['model'] not in definitions:raise ValueError('study requires a public executable model')
    descriptor=json.loads(subprocess.check_output([str(solver),'--describe'],text=True,timeout=60))
    with tempfile.TemporaryDirectory(prefix='atlas-refinement-') as directory:
        generate(directory);current=json.loads((Path(directory)/'models.json').read_text())
    if descriptor['sha256']!=current['sha256']:raise ValueError('study solver and current model catalogue differ')
    canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(version=VERSION,kind='recorded-path-refinement',study=copy.deepcopy(config),
                study_sha256=hashlib.sha256(canonical(config)).hexdigest(),solver_sha256=digest(solver),
                catalogue_hash=descriptor['sha256'],policy=dict(interpolation='explicit piecewise-linear absolute control targets',
                comparison_grid='original observations in supplied order',comparator='requested finest level; no exact-solution claim',
                failure='retain each response; no error metric on failed paths; no replacement finest',
                transient_fields='reported separately; excluded from persistent-state norm'),cases=[])
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    for case in config['cases']:
        base=case['specification'];model=definitions[base['model']]
        row=dict(id=case['id'],model=base['model'],model_sha256=digest(ROOT/'models'/(base['model']+'.json')),levels=[])
        report['cases'].append(row)
        for factor in config['factors']:
            request,indices=refine_recorded(base,factor)
            level=dict(factor=factor,original_indices=indices,comparison_to_finest=None,comparison_to_previous=None)
            row['levels'].append(level)
            try:
                response=execute_recorded(request,solver)
                if response.get('catalogue_hash') and response['catalogue_hash']!=report['catalogue_hash']:raise RuntimeError('catalogue changed during refinement study')
                level['response']=response
            except Exception as error:
                # Transport/process failures are evidence too; continue other
                # independent levels without inventing a state or a prefix.
                level['transport_error']=f'{type(error).__name__}: {error}'
            save()
        first=next((v['response'] for v in row['levels'] if 'parameters' in v.get('response',{})),None)
        if first is None:row.update(fields=None,finest_complete=False);save();continue
        fields=state_layout(model,first['parameters']);row['fields']=fields
        for level in row['levels']:
            r=level.get('response',{})
            if 'parameters' in r and (r['parameters']!=first['parameters'] or r['initial']!=first['initial']):raise ValueError('parameters or initial state changed between refinement levels')
        def observations(level):
            r=level.get('response',{});indices=level['original_indices']
            if not r.get('ok') or len(r.get('points',[]))!=len(r['specification']['path']['knots']):return None
            return [r['points'][i] for i in indices]
        finest=observations(row['levels'][-1]);row['finest_complete']=finest is not None
        previous=None
        for level in row['levels']:
            points=observations(level)
            if points is not None and finest is not None:level['comparison_to_finest']=difference(points,finest,fields)
            if points is not None and previous is not None:level['comparison_to_previous']=difference(points,previous,fields)
            previous=points  # A failed intermediate run breaks adjacency.
        save()
    report['execution_complete']=True;save();return report
