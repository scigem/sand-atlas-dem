import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root=Path('/private/tmp/sand-atlas-dem-walls')
out=root/'evidence/contact-local-frame-integration-v1'
out.mkdir(exist_ok=False)
paths=[]
for directory in ['src','tests','examples','studies','cases']:
 paths.extend(p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in {'.py','.slang','.json'})
paths.append(root/'pyproject.toml')
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(out/'prospective-freeze.json').write_text(json.dumps(dict(files=hashes),indent=2)+'\n')
shutil.copy2(__file__,out/'run.py')
env=dict(os.environ,PYTHONPATH=str(root/'src'),SAND_DEM_METAL='1',MPLCONFIGDIR='/private/tmp/sand-dem-mpl')
with (out/'pytest.log').open('w') as stream:
 result=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
unchanged=all(hashlib.sha256((root/p).read_bytes()).hexdigest()==v for p,v in hashes.items())
(out/'report.json').write_text(json.dumps(dict(identity=out.name,returncode=result.returncode,source_files=len(hashes),sources_unchanged=unchanged,backend='metal',passed=result.returncode==0 and unchanged,scope='Complete current pytest suite with explicit serial Metal; environment-dependent skips retained in log. No new Atlas binary build or material admission.'),indent=2)+'\n')
print((out/'pytest.log').read_text()[-3500:])
raise SystemExit(result.returncode if unchanged else 2)
