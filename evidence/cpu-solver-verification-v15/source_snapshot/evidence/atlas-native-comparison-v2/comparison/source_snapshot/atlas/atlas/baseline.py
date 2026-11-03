"""Capture a software baseline from a fresh export of a pinned Git revision.

This is build/test provenance, not model review or experimental validation.
No commands from an existing manifest are executed by the verifier.
"""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
CHECK_IDS = (
    'definitions', 'configure-native', 'build-native', 'native-tests',
    'elasticity-closed-forms', 'elasticity-evidence-integrity',
    'configure-wasm', 'build-wasm', 'browser-experiment-tests',
    'public-parity', 'time-parity', 'schedule-parity', 'terracotta-parity',
    'package-website', 'catalogue-consistency')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def artifact(root, path):
    path = Path(path)
    return dict(path=path.relative_to(root).as_posix(), sha256=digest(path))


def safe_file(root, item):
    if set(item) != {'path', 'sha256'} or not re.fullmatch('[0-9a-f]{64}', item.get('sha256', '')):
        raise ValueError('Invalid hashed artifact')
    name = Path(item['path'])
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('Artifact paths must be relative without parent traversal')
    target = (root / name).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError('Artifact must be a file within the baseline directory')
    if digest(target) != item['sha256']:
        raise ValueError('Artifact missing or changed: ' + item['path'])
    return target


def verify(path):
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    required = {'schema_version', 'kind', 'id', 'created_at', 'source_commit', 'git_tree',
        'scope', 'outcome', 'versions', 'toolchain', 'configuration', 'inventory', 'checks', 'catalogues',
        'build_products', 'reports', 'limitations', 'reproduction'}
    if set(record) != required or record['schema_version'] != '0.1' or record['kind'] != 'software-baseline':
        raise ValueError('Unsupported baseline record structure')
    for name in ('source_commit', 'git_tree'):
        if not re.fullmatch('[0-9a-f]{40}', record[name]):
            raise ValueError('Baseline must pin complete Git commit/tree identities')
    if record['outcome'] not in ('passed', 'failed') or not record['limitations']:
        raise ValueError('Baseline must declare an outcome and limitations')
    root = path.parent
    inventory = json.loads(safe_file(root, record['inventory']).read_text())
    paths = [item['path'] for item in inventory]
    if len(set(paths)) != len(paths) or not paths:
        raise ValueError('Source inventory must contain unique paths')
    for item in inventory:
        name = Path(item['path'])
        if name.is_absolute() or '..' in name.parts or not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise ValueError('Invalid source inventory entry')
    ids = [item['id'] for item in record['checks']]
    if len(set(ids)) != len(ids) or not set(ids) <= set(CHECK_IDS):
        raise ValueError('Unknown or repeated baseline check')
    for item in record['checks']:
        safe_file(root, item['log'])
        if type(item['returncode']) is not int or not isinstance(item['command'], list):
            raise ValueError('Invalid check result')
    for item in record['reports']:
        safe_file(root, item)
    if record['outcome'] == 'passed':
        if ids != list(CHECK_IDS) or any(item['returncode'] != 0 for item in record['checks']):
            raise ValueError('Passing baseline requires the complete successful check suite')
        if set(record['toolchain']) != {'python','cmake','node','emscripten','platform','native_cxx'}:
            raise ValueError('Passing baseline requires the actual build toolchain')
        if set(record['configuration']) != {'native','wasm'}:
            raise ValueError('Passing baseline requires native/Wasm build configuration')
        if set(record['versions']) != {'python_package','model_schema_versions','compiler','engine','expression_language'}:
            raise ValueError('Passing baseline requires declared version identities')
        products = {'build/atlas_solver','build/atlas_time_solver','build/atlas_research_solver',
            'build-wasm/atlas.js','build-wasm/atlas.wasm','build-wasm/time-fixtures.js',
            'build-wasm/time-fixtures.wasm','build-wasm/research.js','build-wasm/research.wasm'}
        if set(record['build_products']) != products or any(not re.fullmatch('[0-9a-f]{64}', value) for value in record['build_products'].values()):
            raise ValueError('Passing baseline requires native/Wasm product hashes')
        reports = {item['path']:item for item in record['reports']}
        if 'elasticity-results.json' not in reports:
            raise ValueError('Missing analytical result report')
        report = json.loads(safe_file(root, reports['elasticity-results.json']).read_text())
        if report.get('passed') is not True or not report.get('cases') or any(c.get('passed') is not True for c in report['cases']):
            raise ValueError('Analytical result report contradicts passing baseline')
        if set(record['catalogues']) != {'public', 'temporal-fixtures', 'research'}:
            raise ValueError('Passing baseline requires all three catalogue scopes')
        inventory_by_path = {item['path']: item['sha256'] for item in inventory}
        for catalogue in record['catalogues'].values():
            if not re.fullmatch('[0-9a-f]{64}', catalogue['sha256']) or not catalogue['models']:
                raise ValueError('Missing catalogue hash or models')
            for model in catalogue['models']:
                if inventory_by_path.get(model['path']) != model['sha256']:
                    raise ValueError('Model hash does not match the source inventory')
    return record



def catalogue_record(source, generated, definitions):
    catalogue = json.loads((source/generated/'models.json').read_text())
    paths = {}
    for path in (source/definitions).glob('*.json'):
        ident = json.loads(path.read_text())['id']
        if ident in paths:
            raise ValueError('Repeated model identity in '+definitions)
        paths[ident] = path
    if set(paths) != {m['id'] for m in catalogue['models']}:
        raise ValueError('Generated/source catalogue identities differ in '+definitions)
    return dict(sha256=catalogue['sha256'], models=[dict(id=m['id'], status=m['status'],
        **artifact(source,paths[m['id']])) for m in catalogue['models']])


def capture(ref, destination):
    # Resolve before creating the output directory; never copy a dirty checkout.
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', '--end-of-options', ref+'^{commit}'], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(['git', 'rev-parse', commit+'^{tree}'], cwd=ROOT, text=True).strip()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'logs').mkdir()
    record = dict(schema_version='0.1', kind='software-baseline', id='baseline-'+commit[:12]+'-'+uuid.uuid4().hex[:12],
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_commit=commit,
        git_tree=tree, scope='Fresh committed-source export; native and WebAssembly software verification. No experimental dataset or manuscript build.',
        outcome='failed', versions={}, toolchain={}, configuration={}, inventory={}, checks=[], catalogues={}, build_products={}, reports=[],
        limitations=['Independent scientific review remains pending for all public formulations.',
            'Five public executable formulations are distinct from discovery candidates, synthetic fixtures and the restricted Terracotta research draft.',
            'No Wichtmann data, calibration, experimental validation, paper compilation or full literature-review claim is included.',
            'Compiler, engine and expression language have source identities rather than separately frozen semantic versions; schema v1 remains open.',
            'Binary hashes identify this build/platform; reproduction on another platform is assessed by declared numerical checks, not bitwise equality.'],
        reproduction=['git checkout --detach '+commit, 'python3 tools/capture_baseline.py --ref '+commit+' --output artifacts/reproduced-baseline'])
    manifest = destination/'manifest.json'
    def save():
        dump(manifest, record)
    def run(ident, command, source, env):
        print('Starting '+ident, flush=True)
        start = time.monotonic()
        try:
            result = subprocess.run(command, cwd=source, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=900)
            code, output = result.returncode, result.stdout
        except subprocess.TimeoutExpired as error:
            code = 124
            output = error.stdout or ''
            if isinstance(output, bytes): output = output.decode(errors='replace')
            output += '\nBaseline command timed out after 900 seconds.\n'
        log = destination/'logs'/(ident+'.txt')
        log.write_text(output)
        record['checks'].append(dict(id=ident, command=command, returncode=code,
            elapsed_seconds=time.monotonic()-start, log=artifact(destination, log)))
        save()
        print(('Passed ' if code == 0 else 'Failed ')+ident, flush=True)
        if code:
            raise RuntimeError(ident+' failed; retained log: '+str(log))
    try:
        archive = subprocess.check_output(['git','archive','--format=tar',commit], cwd=ROOT)
        with tempfile.TemporaryDirectory(prefix='atlas-baseline-') as directory:
            # SDK cache builds derive relative paths from their working directory.
            # Resolve macOS /var -> /private/var before handing paths to tools.
            directory = Path(directory).resolve()
            source = directory/'source'
            source.mkdir()
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(source, filter='data')
            # A later controller must not silently redefine an older baseline.
            for name in ('atlas/baseline.py','tools/capture_baseline.py'):
                if not (source/name).is_file() or digest(source/name) != digest(ROOT/name):
                    raise ValueError('Run the baseline controller from the target revision: '+name)
            inventory = [artifact(source,p) for p in sorted(source.rglob('*')) if p.is_file()]
            dump(destination/'source-inventory.json', inventory)
            record['inventory'] = artifact(destination, destination/'source-inventory.json')
            env = os.environ.copy()
            # Use fresh build/cache state, rather than the shared checkout cache.
            env['EM_CACHE'] = str(Path(directory)/'emscripten-cache')
            for tool in ('cmake','node','emcc','emcmake'):
                if not shutil.which(tool): raise ValueError('Missing required tool: '+tool)
            for tool,command in [('python',[sys.executable,'--version']),('cmake',['cmake','--version']),('node',['node','--version']),('emscripten',['emcc','--version'])]:
                record['toolchain'][tool] = subprocess.check_output(command, text=True, env=env).strip()
            record['toolchain']['platform'] = platform.platform()
            package = subprocess.check_output([sys.executable,'-c','import atlas; print(atlas.__version__)'], cwd=source, text=True).strip()
            schemas = sorted({json.loads(p.read_text())['schema_version'] for directory,pattern in [('models','*.json'),('models/research','*.json'),('tests/fixtures/time','*.json')] for p in (source/directory).glob(pattern)})
            record['versions'] = dict(python_package=package, model_schema_versions=schemas,
                compiler='Source-pinned in source inventory; no separate frozen semantic version',
                engine='Source-pinned in source inventory; no separate frozen semantic version',
                expression_language='Source-pinned in source inventory; no separate frozen semantic version')
            save()
            commands = [
                ('definitions',[sys.executable,'-m','atlas.cli','validate']),
                ('configure-native',['cmake','-S','.','-B','build','-DCMAKE_BUILD_TYPE=Release']),
                ('build-native',['cmake','--build','build','--parallel','2']),
                ('native-tests',['ctest','--test-dir','build','--verbose']),
                ('elasticity-closed-forms',[sys.executable,'tools/verify_elasticity.py','--output','artifacts/elasticity-results.json']),
                ('elasticity-evidence-integrity',[sys.executable,'-m','atlas.cli','evidence','catalogue/evidence/linear-elastic-analytical-v1.json']),
                ('configure-wasm',['emcmake','cmake','-S','.','-B','build-wasm','-DCMAKE_BUILD_TYPE=Release']),
                ('build-wasm',['cmake','--build','build-wasm','--parallel','2']),
                ('browser-experiment-tests',['node','tools/test_experiments.mjs']),
                ('public-parity',['node','tools/test_wasm.mjs']),
                ('time-parity',['node','tools/test_time_wasm.mjs']),
                ('schedule-parity',['node','tools/test_schedule_wasm.mjs']),
                ('terracotta-parity',['node','tools/test_terracotta_wasm.mjs']),
                ('package-website',[sys.executable,'tools/build_web.py'])]
            for ident,command in commands:
                run(ident, command, source, env)
                if ident == 'elasticity-closed-forms':
                    shutil.copyfile(source/'artifacts/elasticity-results.json', destination/'elasticity-results.json')
                    record['reports'].append(artifact(destination,destination/'elasticity-results.json'))
                    save()
            # Capture the actual configured compiler, rather than assuming $CXX.
            cache = (source/'build/CMakeCache.txt').read_text()
            compiler = next(line.split('=',1)[1] for line in cache.splitlines() if line.startswith('CMAKE_CXX_COMPILER:FILEPATH='))
            record['toolchain']['native_cxx'] = subprocess.check_output([compiler,'--version'],text=True).strip()
            settings = {'CMAKE_BUILD_TYPE','CMAKE_CXX_COMPILER','CMAKE_CXX_FLAGS','CMAKE_CXX_FLAGS_RELEASE',
                'CMAKE_EXE_LINKER_FLAGS','CMAKE_EXE_LINKER_FLAGS_RELEASE','CMAKE_GENERATOR','CMAKE_TOOLCHAIN_FILE'}
            for scope,folder in [('native','build'),('wasm','build-wasm')]:
                values = {}
                for line in (source/folder/'CMakeCache.txt').read_text().splitlines():
                    if '=' not in line or ':' not in line.split('=',1)[0]: continue
                    key = line.split(':',1)[0]
                    if key in settings: values[key] = line.split('=',1)[1]
                record['configuration'][scope] = values
            scopes = [('public','build/generated','models'),('temporal-fixtures','build/time-generated','tests/fixtures/time'),('research','build/research-generated','models/research')]
            for scope,generated,definitions in scopes:
                record['catalogues'][scope] = catalogue_record(source,generated,definitions)
            # Read-only consistency assertions supplement the parity executables.
            consistency = [sys.executable,'-c',
                "import json; from pathlib import Path; a=json.loads(Path('build/generated/models.json').read_text()); b=json.loads(Path('build-wasm/generated/models.json').read_text()); c=json.loads(Path('web/dist/models.json').read_text()); assert a==b==c; x=json.loads(Path('catalogue/manifest.json').read_text()); y=json.loads(Path('web/dist/catalogue.json').read_text()); assert x==y; assert {m['id'] for m in a['models']}=={m['id'] for m in x['models'] if m['executable']}; print('Native, Wasm, packaged definitions and public executable identities agree.')"]
            run('catalogue-consistency',consistency,source,env)
            for name in ['build/atlas_solver','build/atlas_time_solver','build/atlas_research_solver','build-wasm/atlas.js','build-wasm/atlas.wasm','build-wasm/time-fixtures.js','build-wasm/time-fixtures.wasm','build-wasm/research.js','build-wasm/research.wasm']:
                record['build_products'][name] = digest(source/name)
            record['outcome'] = 'passed'
            save()
            verify(manifest)
    except BaseException:
        error_log = destination/'controller-error.txt'
        error_log.write_text(traceback.format_exc())
        record['reports'].append(artifact(destination,error_log))
        record['outcome'] = 'failed'
        save()
        raise
    return manifest
