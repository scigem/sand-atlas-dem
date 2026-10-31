"""One-shot dependency runner; never restart a missing reference process."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('/Users/bmar5496/code/sand-atlas-dem')
launch_path=Path('/private/tmp/hostun-minimum-gpu-launch-v1.json')
frozen=launch_path.read_bytes()
launch=json.loads(frozen)
policy=launch['prospective_gpu_policy']
reference=root/policy['reference_report']
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
while not reference.exists():
    try:
        os.kill(launch['live_reference_pid'],0)
    except ProcessLookupError:
        raise RuntimeError('Reference process disappeared without its final report; no retry launched')
    time.sleep(10)
# The report writer atomically replaces files; a present report is complete.
report=json.loads(reference.read_bytes())
if not report['passed'] or report['protocol_sha256']!=launch['expected_reference_protocol_sha256']:
    raise RuntimeError('Reference failed or differs from the prospectively frozen protocol')
if launch_path.read_bytes()!=frozen or digest(root/'examples/mesh_quadrature_study.py')!=launch['expected_script_sha256']:
    raise RuntimeError('Launch inputs changed while waiting')
if any(digest(root/p)!=h for p,h in launch['source_hashes'].items()):
    raise RuntimeError('Solver changed while waiting')
policy['reference_report_sha256']=digest(reference)
protocol=root/launch['resolved_protocol']
with protocol.open('x') as handle:
    json.dump(policy,handle,indent=2)
    handle.write('\n')
environment=os.environ.copy()
environment['PYTHONPATH']='src'
completed=subprocess.run([str(root/'.venv/bin/python'),'examples/mesh_quadrature_study.py',str(protocol),'--output',launch['output']],cwd=root,env=environment,check=False)
raise SystemExit(completed.returncode)
