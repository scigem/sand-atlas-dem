"""Validate and encode ordered physical-time control histories for the C++ ABI.

Constitutive integration and mixed-control solving live in the common engine.
The flat CSV is a versioned transport layout, not an alternative model language.
"""
import json
import math
from .model import ROOT
from .structure import check
from .dsl import ValidationError

DEFAULTS={'method':'backward-euler-step-doubling','relative_tolerance':1e-6,
          'absolute_scale_fraction':1e-9,'minimum_step':1e-12,'maximum_step':1e6,
          'maximum_substeps':20000,'nonlinear_tolerance':1e-10,'maximum_iterations':40}

def validate_schedule(spec):
    check(spec,json.loads((ROOT/'schemas/run-v0.2.schema.json').read_text()))
    settings={**DEFAULTS,**spec['solver']}
    if settings['minimum_step']>settings['maximum_step']:
        raise ValidationError('minimum step exceeds maximum step')
    seen=set();time=spec['start_time'];total=0
    for segment in spec['segments']:
        if segment['id'] in seen:raise ValidationError('duplicate segment ID')
        seen.add(segment['id']);next_time=time+segment['duration']
        if not math.isfinite(next_time) or not next_time>time:
            raise ValidationError('segment endpoint is not a finite, representably later time')
        # Every requested output interval must itself be representable.
        previous=time
        for i in range(1,segment['samples']+1):
            sample=next_time if i==segment['samples'] else time+segment['duration']*i/segment['samples']
            if not sample>previous:raise ValidationError('output sample times are not increasing')
            previous=sample
        total+=segment['samples'];time=next_time
    if total>10000:raise ValidationError('schedule exceeds 10000 regular output samples')
    return settings


def schedule_command(spec,solver):
    settings=validate_schedule(spec)
    csv=lambda values:','.join(format(v,'.17g') for v in values)
    wire=[]
    for segment in spec['segments']:
        wire.extend([segment['duration'],segment['samples']])
        for control in segment['controls']:
            wire.extend([int(control['quantity']=='stress'),{'hold':0,'rate':1,'target':2}[control['kind']],control.get('value',0)])
    command=[str(solver),'--mode','schedule','--model',spec['model'],'--start-time',format(spec['start_time'],'.17g'),
             '--segments',csv(wire),'--segment-ids',','.join(s['id'] for s in spec['segments'])]
    for field in ['parameters','initial','initial_strain','initial_strain_rate']:
        if field in spec:command.extend(['--'+field.replace('_','-'),csv(spec[field])])
    for key,option in [('relative_tolerance','rtol'),('absolute_scale_fraction','atol'),('minimum_step','min-time-step'),('maximum_step','max-time-step'),('maximum_substeps','max-substeps'),('nonlinear_tolerance','nonlinear-tolerance'),('maximum_iterations','max-iterations')]:
        command.extend(['--'+option,str(settings[key])])
    return command
