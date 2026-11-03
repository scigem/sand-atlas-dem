import argparse
import json
from pathlib import Path
import subprocess
from .model import ROOT, all_models


def main():
    parser=argparse.ArgumentParser(description='Constitutive Atlas research and simulation tools')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('validate',help='Validate every executable definition')
    migrate=sub.add_parser('migrate-model',help='Write an explicit v0.3 formulation without changing the source')
    migrate.add_argument('definition',type=Path)
    migrate.add_argument('--output',type=Path,required=True)
    migrate.add_argument('--stress-scope',choices=('effective','total'))
    evidence=sub.add_parser('evidence',help='Validate numerical evidence integrity; does not confer review status');evidence.add_argument('record',type=Path);evidence.add_argument('--current',action='store_true')
    generate=sub.add_parser('generate');generate.add_argument('destination',type=Path)
    db=sub.add_parser('database');db.add_argument('destination',type=Path)
    ingest=sub.add_parser('ingest');ingest.add_argument('archive',type=Path);ingest.add_argument('--output',type=Path,default=ROOT/'data/normalized');ingest.add_argument('--metadata',type=Path)
    discovery=sub.add_parser('discover');discovery.add_argument('--query',required=True);discovery.add_argument('--output',required=True,type=Path);discovery.add_argument('--rows',type=int,default=50);discovery.add_argument('--cursor',default='*')
    run=sub.add_parser('run');run.add_argument('specification',type=Path);run.add_argument('--solver',type=Path,default=ROOT/'build/atlas_solver');run.add_argument('--output',type=Path,required=True)
    replay=sub.add_parser('replay',help='Replay a validated ordered recorded path');replay.add_argument('specification',type=Path);replay.add_argument('--solver',type=Path,default=ROOT/'build/atlas_solver');replay.add_argument('--output',type=Path,required=True)
    initializer=sub.add_parser('initialize',help='Resolve a typed initialization recipe');initializer.add_argument('recipe',type=Path);initializer.add_argument('configuration',type=Path);initializer.add_argument('--model-definition',type=Path,required=True);initializer.add_argument('--output',type=Path,required=True)
    prepare=sub.add_parser('prepare-replay',help='Convert normalized drained observations with explicit assumptions');prepare.add_argument('experiment',type=Path);prepare.add_argument('configuration',type=Path);prepare.add_argument('--output',type=Path,required=True)
    refine=sub.add_parser('refine-recorded',help='Investigate ordered caller-path resolution');refine.add_argument('configuration',type=Path);refine.add_argument('--solver',type=Path,default=ROOT/'build/atlas_solver');refine.add_argument('--output',type=Path,required=True)
    benchmark=sub.add_parser('benchmark');benchmark.add_argument('configuration',type=Path);benchmark.add_argument('--solver',type=Path,default=ROOT/'build/atlas_solver');benchmark.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='validate':print(f'{len(all_models())} declarative models passed structural and semantic validation')
    elif args.command=='migrate-model':
        from .model import load, migrate_formulation
        model=migrate_formulation(load(args.definition),stress_scope=args.stress_scope)
        with args.output.open('x') as stream:
            stream.write(json.dumps(model,indent=2,allow_nan=False)+'\n')
        print(args.output)
    elif args.command=='evidence':
        from .evidence import validate_record
        record=validate_record(args.record,current=args.current);print(f"{record['id']}: record verified ({record.get('outcome',record.get('status'))}); no automatic model promotion")
    elif args.command=='generate':
        from .compiler import generate
        generate(args.destination);print(args.destination)
    elif args.command=='database':
        from .research import export_database
        print(export_database(args.destination))
    elif args.command=='ingest':
        from .data import ingest
        metadata=json.loads(args.metadata.read_text()) if args.metadata else None
        outputs=ingest(args.archive,args.output,metadata);print(f'Normalized {len(outputs)} experiments into {args.output}')
    elif args.command=='discover':
        from .research import discover
        result=discover(args.query,args.output,args.rows,args.cursor);print(f"Saved {len(result['response']['message']['items'])} unscreened records to {args.output}")
    elif args.command=='run':
        spec=json.loads(args.specification.read_text());spec=spec.get('specification',spec)
        if spec.get('schema_version')=='0.2':
            from .schedule import schedule_command
            command=schedule_command(spec,args.solver)
        else:
            from .structure import check
            check(spec,json.loads((ROOT/'schemas/run.schema.json').read_text()))
            known={'model','parameters','initial','protocol','target','steps','cycles'}
            if set(spec)-known or not {'model','protocol','target','steps','cycles'}<=spec.keys():raise ValueError('invalid run specification fields')
            command=[str(args.solver)]
            for key,value in spec.items():command.extend(['--'+key,','.join(str(x) for x in value) if isinstance(value,list) else str(value)])
        result=subprocess.run(command,text=True,capture_output=True,timeout=60);data=json.loads(result.stdout)
        if spec.get('schema_version')=='0.2':data['specification']=spec
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(data,indent=2)+'\n')
        print(f"{'Completed' if data['ok'] else 'Failed'}: {args.output}")
        if not data['ok']:raise SystemExit(1)
    elif args.command=='replay':
        from .recorded import execute_recorded
        spec=json.loads(args.specification.read_text())
        result=execute_recorded(spec,args.solver)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(f"{'Completed' if result['ok'] else 'Failed'}: {args.output}")
        if not result['ok']:raise SystemExit(1)
    elif args.command=='initialize':
        import hashlib
        from .initialization import initialize
        from .experiment_replay import parse_json
        paths={'recipe':args.recipe,'configuration':args.configuration,'model':args.model_definition}
        if args.output.resolve() in {p.resolve() for p in paths.values()}:
            raise ValueError('initialization output must not overwrite an input')
        raw={k:p.read_bytes() for k,p in paths.items()}
        result=initialize(parse_json(raw['recipe']),parse_json(raw['configuration']),parse_json(raw['model']))
        result['provenance']['input_file_sha256']={k:hashlib.sha256(v).hexdigest() for k,v in raw.items()}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(f'Initialized {result["run_fields"]["model"]}: {args.output}')
    elif args.command=='prepare-replay':
        import hashlib
        from .experiment_replay import prepare_replay, parse_json
        provenance_path=args.output.with_suffix(args.output.suffix+'.provenance.json')
        inputs={args.experiment.resolve(),args.configuration.resolve()}
        if args.output.resolve() in inputs or provenance_path.resolve() in inputs:
            raise ValueError('output must not overwrite experiment or configuration')
        experiment_bytes=args.experiment.read_bytes()
        configuration_bytes=args.configuration.read_bytes()
        bundle=prepare_replay(parse_json(experiment_bytes),parse_json(configuration_bytes))
        bundle['provenance']['input_file_sha256']={
            'experiment':hashlib.sha256(experiment_bytes).hexdigest(),
            'configuration':hashlib.sha256(configuration_bytes).hexdigest()}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(bundle['request'],indent=2,allow_nan=False)+'\n')
        provenance_path.write_text(json.dumps(bundle['provenance'],indent=2,allow_nan=False)+'\n')
        print(f'Prepared {len(bundle["request"]["path"]["knots"])} ordered observations: {args.output}; provenance: {provenance_path}')
    elif args.command=='refine-recorded':
        from .refinement import execute_study
        report=execute_study(json.loads(args.configuration.read_text()),args.solver,args.output)
        levels=[level for case in report['cases'] for level in case['levels']]
        successful=sum(bool(level.get('response',{}).get('ok')) for level in levels)
        print(f'Recorded {successful}/{len(levels)} successful refinement runs: {args.output}')
        if successful!=len(levels):raise SystemExit(1)
    elif args.command=='benchmark':
        from .benchmark import execute
        result=execute(args.configuration,args.solver,args.output);print(f"Recorded {len(result['comparisons'])} comparisons: {args.output}")


if __name__=='__main__':main()
