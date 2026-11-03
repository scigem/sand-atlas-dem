"""Reproducible monotonic curve comparisons; no extrapolation or hidden fitting."""
import bisect
import hashlib
import json
import math
from pathlib import Path
import subprocess
from .model import ROOT, all_models, defaults


def interpolate(points,x):
    xs=[p[0] for p in points]
    if x<xs[0]-1e-12 or x>xs[-1]+1e-12:raise ValueError('extrapolation is forbidden')
    i=bisect.bisect_right(xs,x)
    if i==0:return points[0][1]
    if i==len(xs):return points[-1][1]
    a,b=points[i-1],points[i]
    return a[1]+(b[1]-a[1])*(x-a[0])/(b[0]-a[0])


def monotonic_channel(rows,key,maximum):
    result=[];duplicates=0
    for r in rows:
        x=r['axial_strain']
        if x>maximum:break
        if x<0:continue
        if result and x<result[-1][0]:raise ValueError('nonmonotonic strain history requires path replay, not sorting')
        if key not in r:raise ValueError('missing measured channel: '+key)
        pair=(x,r[key])
        if result and x==result[-1][0]:result[-1]=pair;duplicates+=1
        else:result.append(pair)
    if len(result)<2:raise ValueError('insufficient monotonic data')
    return result,duplicates


def curve_metrics(measured,predicted,samples=101):
    lo=max(measured[0][0],predicted[0][0]);hi=min(measured[-1][0],predicted[-1][0])
    if hi<=lo:raise ValueError('curves have no overlapping strain interval')
    residual=[];observed=[]
    for i in range(samples):
        x=lo+(hi-lo)*i/(samples-1);y=interpolate(measured,x);yp=interpolate(predicted,x)
        residual.append(yp-y);observed.append(y)
    rmse=math.sqrt(sum(v*v for v in residual)/samples);spread=max(observed)-min(observed)
    return dict(rmse=rmse,mae=sum(abs(v) for v in residual)/samples,max_abs=max(abs(v) for v in residual),
                normalized_rmse=rmse/spread if spread>0 else None,normalization='observed range on evaluation grid',
                evaluation_grid=dict(kind='uniform axial strain',samples=samples,start=lo,end=hi))


def execute(configuration,solver,output):
    configuration=Path(configuration);config=json.loads(configuration.read_text())
    if config.get('schema_version')!='0.1':raise ValueError('unsupported benchmark version')
    parameter_path=ROOT/config['parameter_set'];parameter_set=json.loads(parameter_path.read_text())
    m=next(m for m in all_models() if m['id']==parameter_set['model']);params=defaults(m)
    if set(parameter_set['parameters'])-params.keys():raise ValueError('unknown parameter')
    params.update(parameter_set['parameters'])
    result=dict(schema_version='0.1',kind='benchmark-report',id=config['id'],phase=config['phase'],claim=config['claim'],
                configuration_sha256=hashlib.sha256(configuration.read_bytes()).hexdigest(),
                parameter_set=parameter_set,parameter_set_sha256=hashlib.sha256(parameter_path.read_bytes()).hexdigest(),
                solver_sha256=hashlib.sha256(Path(solver).read_bytes()).hexdigest(),comparisons=[])
    for test in config['experiments']:
        path=ROOT/test['path'];experiment=json.loads(path.read_text());record=dict(experiment=experiment['id'],split=test['split'],experiment_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_sha256=experiment['provenance']['member_sha256'])
        if experiment['protocol']!='drained-triaxial':raise ValueError('this benchmark driver currently supports monotonic drained tests only')
        measured,duplicate_count=monotonic_channel(experiment['rows'],'q',config['max_axial_strain'])
        z=[]
        for field in m['state']:
            if field['name']=='sigma':z+=experiment['initial']['effective_stress']
            elif field['name']=='e':z.append(experiment['initial']['void_ratio'])
            elif field.get('shape')=='sym':z+=field['initial']
            else:z.append(field['initial'])
        spec=dict(model=m['id'],parameters=[params[n] for n in m['parameters']],initial=z,protocol=1,target=measured[-1][0],steps=config['steps'],cycles=0)
        command=[str(solver)]
        for k,v in spec.items():command+=['--'+k,','.join(map(str,v)) if isinstance(v,list) else str(v)]
        process=subprocess.run(command,capture_output=True,text=True,timeout=60);run=json.loads(process.stdout)
        record.update(ok=run['ok'],message=run.get('message',''),specification=spec,catalogue_hash=run.get('catalogue_hash'),
                      duplicate_strain_policy='last observation at identical strain, preserving record order',duplicate_strain_count=duplicate_count)
        if run['ok']:
            record['q_metrics']=curve_metrics(measured,[(p['strain'][0],p['q_signed']) for p in run['points']])
            volume,_=monotonic_channel(experiment['rows'],'volumetric_strain',config['max_axial_strain'])
            record['volumetric_metrics']=curve_metrics(volume,[(p['strain'][0],sum(p['strain'][:3])) for p in run['points']])
        # Full runs are retained locally for plotting; raw data are never embedded.
        record['simulation']=run;result['comparisons'].append(record)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
