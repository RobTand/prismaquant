"""Provision a scoped CPU test interpreter without mutating the fleet base venv."""
from pathlib import Path
import json
import subprocess
import sys
import venv

base=Path('/home/rob/venvs/pq-cpu312')
target=Path('/home/rob/venvs/pq-cpu312-glm-best-form-b1eb1dccc')
source=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/producer-source-best-tile-reviewed-01')
assert sys.prefix==str(base),sys.prefix
if target.exists():
    raise RuntimeError(f'scoped environment already exists; inspect before reusing: {target}')
venv.EnvBuilder(with_pip=False).create(target)
site=next((target/'lib').glob('python*/site-packages'))
base_site=base/'lib'/site.parent.name/'site-packages'
assert base_site.is_dir()
(site/'fleet-base-dependencies.pth').write_text(str(base_site)+'\n')
python=str(target/'bin/python')
command=[sys.executable,'tools/provision_tessera_pin.py','--python',python,'--clone',str(source)]
subprocess.run(command,check=True)
subprocess.run([*command,'--check-only'],check=True)
probe=subprocess.check_output([python,'-I','-c','import json,tessera,torch; print(json.dumps(dict(tessera=tessera.__file__,torch=torch.__version__)))'],text=True)
identity=json.loads(probe);assert identity['tessera'].startswith(str(site)+'/')
print(json.dumps(dict(status='SCOPED_ENV_READY',python=python,base_dependencies=str(base_site),identity=identity)))
