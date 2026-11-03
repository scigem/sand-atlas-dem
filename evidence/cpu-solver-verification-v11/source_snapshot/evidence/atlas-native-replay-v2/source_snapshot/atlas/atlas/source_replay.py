"""Portable source-only replay of explicitly registered trusted project code.
Explicit trusted Python entrypoints only. No source discovery or pin substitution.
"""
import base64,gzip,hashlib,io,json,os,pathlib,subprocess,sys,tempfile
VERSION='atlas-source-replay-capsule-0.1'

def digest(blob):return hashlib.sha256(blob).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def relative(name):
 require(type(name)is str and '\\' not in name and name!='','relative POSIX path required')
 p=pathlib.PurePosixPath(name)
 require(not p.is_absolute() and all(x not in ('.','..','') for x in name.split('/')),'unsafe source path')
 return name

def export(root,pins,entry,args=(),expected_exit=0,overrides=None):
 """overrides supply historically resolved source bytes, checked against original pins.
 In production resolve overrides via document-bound checked_receipt, not current hashes.
 """
 root=pathlib.Path(root).resolve();overrides=overrides or {};require(type(pins)is dict and pins,'explicit closure required')
 files={}
 for name,pin in sorted(pins.items()):
  relative(name);target=(root/name).resolve();require(target.is_relative_to(root),'source escape')
  blob=overrides[name] if name in overrides else target.read_bytes()
  require(type(blob)is bytes and digest(blob)==pin,'original source pin mismatch: '+name)
  files[name]={'sha256':pin,'bytes':len(blob),'base64':base64.b64encode(blob).decode('ascii')}
 c={'version':VERSION,'closure':'explicit source/data files, stdlib Python interpreter external','entrypoint':entry,'args':list(args),'expected_exit':expected_exit,'files':files,'claims':{'compiled_rebuild':False,'scientific_rerun':False,'environment_bit_reproduction':False}}
 validate(c);return c

def validate(c):
 require(type(c)is dict and c.get('version')==VERSION,'capsule version')
 require(type(c.get('files'))is dict and 0<len(c['files'])<=1024,'bounded closure')
 total=0
 for name,pin in c['files'].items():
  relative(name);require(type(pin)is dict and set(pin)=={'sha256','bytes','base64'},'source blob shape')
  require(type(pin['sha256'])is str and len(pin['sha256'])==64 and all(x in '0123456789abcdef' for x in pin['sha256']),'source digest')
  require(type(pin['bytes'])is int and 0<=pin['bytes']<=16*1024*1024,'source size')
  require(type(pin['base64'])is str,'source encoding')
  b=base64.b64decode(pin['base64'],validate=True);require(len(b)==pin['bytes'] and digest(b)==pin['sha256'],'source blob integrity')
  total+=len(b);require(total<=64*1024*1024,'capsule total size')
  for parent in pathlib.PurePosixPath(name).parents:
   require(parent.as_posix() not in c['files'],'file/directory collision')
 require(relative(c['entrypoint']) in c['files'] and c['entrypoint'].endswith('.py'),'archived Python entrypoint required')
 require(type(c['args'])is list and all(type(x)is str and '\x00' not in x for x in c['args']),'literal command args')
 require(type(c['expected_exit'])is int and 0<=c['expected_exit']<=255,'expected process exit')
 return c

def materialize(c,destination):
 validate(c);d=pathlib.Path(destination).resolve();require(not d.exists(),'fresh capsule tree required');d.mkdir(parents=True)
 for name,pin in c['files'].items():
  p=d/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(pin['base64'],validate=True))
 return d

def replay(c,timeout=30):
 """Re-audit original source bytes in a fresh process/tree; no ambient repo fallback.
 Runtime/library version is recorded. Caller interprets output against frozen gates.
 """
 validate(c)
 with tempfile.TemporaryDirectory(prefix='atlas-frozen-replay-') as tmp:
  root=materialize(c,pathlib.Path(tmp)/'repo')
  args=[x.replace('{root}',str(root)) for x in c['args']]
  bootstrap='import os,sys,runpy; root=sys.argv[1]; entry=sys.argv[2]; sys.path.insert(0,root); os.chdir(root); sys.argv=[entry]+sys.argv[3:]; runpy.run_path(entry,run_name="__main__")'
  env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','ATLAS_'))};env['PYTHONDONTWRITEBYTECODE']='1'
  process=subprocess.run([sys.executable,'-I','-S','-B','-c',bootstrap,str(root),str(root/c['entrypoint']),*args],cwd=root,env=env,capture_output=True,timeout=timeout)
  for name,pin in c['files'].items():require(digest((root/name).read_bytes())==pin['sha256'],'archived input changed during replay: '+name)
  return {'version':'atlas-source-replay-result-0.1','capsule_sha256':digest(json.dumps(c,sort_keys=True,separators=(',',':')).encode()),'returncode':process.returncode,'expected_exit':c['expected_exit'],'expected_exit_matched':process.returncode==c['expected_exit'],'stdout':process.stdout.decode(errors='strict'),'stderr':process.stderr.decode(errors='strict'),'stdout_sha256':digest(process.stdout),'stderr_sha256':digest(process.stderr),'python':sys.version,'execution_scope':'trusted Python entrypoint; side effects are not sandboxed','note':'Runs the trusted archived Python entrypoint. Not a sandbox or compiled/scientific rerun.'}


REGISTRY_VERSION = 'atlas-registered-source-replay-0.1'


def strict_json(blob):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON member: ' + key)
            result[key] = value
        return result
    return json.loads(blob, object_pairs_hook=unique)


def pinned_file(root, pin):
    name = relative(pin['path'])
    target = root / name
    require(not any(p.is_symlink() for p in [target, *target.parents] if p != root.parent),
            'symlink in replay input path')
    require(target.resolve().is_relative_to(root), 'replay input escape')
    blob = target.read_bytes()
    require(len(blob) == pin['bytes'] and digest(blob) == pin['sha256'],
            'registered replay input changed: ' + name)
    return blob


def registered(root, registry, replay_id, timeout=30):
    """Verify original documents, run their registered closure and match exact output.

    The tracked registry is the trust anchor. Never accept an unreviewed registry
    from an external party: this runs trusted Python code, not sandboxed code.
    A successful replay can reproduce a negative finding; it is not model admission.
    """
    root = pathlib.Path(root).resolve()
    registry_path = root / relative(registry)
    require(registry_path.resolve().is_relative_to(root), 'registry escape')
    spec = strict_json(registry_path.read_bytes())
    require(spec.get('version') == REGISTRY_VERSION, 'registered replay version')
    record = spec['replays'][replay_id]
    held = [(pin, pinned_file(root, pin)) for pin in record['documents']]
    capsule_bytes = pinned_file(root, record['capsule'])
    encoding = record.get('capsule_encoding', 'json')
    require(encoding in ('json', 'json-gzip'), 'unsupported capsule encoding')
    decoded = capsule_bytes
    if encoding == 'json-gzip':
        # Capsule payloads allow at most64MiB source bytes; base64 plus metadata
        # must fit this separate96MiB decoded-document ceiling.
        with gzip.GzipFile(fileobj=io.BytesIO(capsule_bytes)) as stream:
            decoded = stream.read(96 * 1024 * 1024 + 1)
        require(len(decoded) <= 96 * 1024 * 1024, 'decoded capsule exceeds limit')
    c = strict_json(decoded)
    validate(c)
    # Bind the archived assessor to the original review's explicit source identity.
    if 'review_document_index' in record:
        review = strict_json(held[record['review_document_index']][1])
        source_pin = c['files'][record['reviewed_source']]['sha256']
        require(source_pin in review['reviewed_source_hashes'], 'source absent from original review')
    bindings = record.get('source_bindings', [])
    require('review_document_index' in record or bindings, 'original source identity binding required')
    for binding in bindings:
        value = strict_json(held[binding['document_index']][1])
        for key in binding['pointer']:
            value = value[key]
        require(value == c['files'][relative(binding['source_path'])]['sha256'],
                'archived source differs from original document binding')
    expected = held[record['expected_stdout_document_index']][1]
    result = replay(c, timeout=timeout)
    require(result['expected_exit_matched'], 'archived process exit changed')
    require(result['stderr'] == '', 'archived replay emitted stderr')
    require(result['stdout'].encode() == expected, 'original witness output changed')
    for pin, blob in held:
        require(pinned_file(root, pin) == blob, 'original document changed during replay')
    require(pinned_file(root, record['capsule']) == capsule_bytes, 'capsule changed during replay')
    result['replay_id'] = replay_id
    result['finding'] = record['finding']
    result['historical_output_reproduced'] = True
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Replay registered trusted project Python sources')
    parser.add_argument('replay_id')
    parser.add_argument('--root', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument('--registry', default='catalogue/source-replays/transport-audit-v1.json')
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    print(json.dumps(registered(args.root, args.registry, args.replay_id, timeout=args.timeout), indent=2, allow_nan=False))
