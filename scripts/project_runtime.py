"""Portable runtime configuration; no host identifiers or global mutations."""
import os,re,json,hashlib,sys,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('CRAFT_DATA_ROOT',str(ROOT/'.artifacts'))).expanduser().resolve()
RUN=os.environ.get('CRAFT_RUN','portable-v2')
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',RUN):
    raise ValueError('CRAFT_RUN must be a simple version name')
MODELS=DATA/'models'/RUN
REPORTS=DATA/'reports'/RUN
LOGS=DATA/'logs'/RUN
CONFIG=ROOT/'configs/train.yaml'
GPU=os.environ.get('CRAFT_GPU','0')
THREADS=int(os.environ.get('CRAFT_THREADS','2'))
if THREADS<1:raise ValueError('CRAFT_THREADS must be positive')
def owned(path):
    path=Path(path).expanduser().resolve()
    if not path.is_relative_to(DATA) or path==DATA:
        raise ValueError('Output must be below CRAFT_DATA_ROOT')
    return path
def initialize():
    for path in (MODELS,REPORTS,LOGS):path.mkdir(parents=True,exist_ok=True)
def affinity():
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:THREADS])
def sources():
    paths=[ROOT/'python.sh']
    for folder,patterns in [('scripts',['*.py']),('tools',['*.py']),('tests',['*.py']),('dreamerv3',['*.py','*.yaml']),('configs',['*.yaml','*.json']),('requirements',['*.txt'])]:
        for pat in patterns:paths.extend((ROOT/folder).rglob(pat))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
def environment():
    return dict(python=sys.version.split()[0],packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']})
def freeze():
    initialize()
    path=REPORTS/'release-lock.json'
    value=dict(version='portable-v2',sources=sources(),environment=environment(),gpu=GPU,threads=THREADS)
    if path.exists():
        if json.loads(path.read_text())!=value:raise RuntimeError('Release changed; select a NEW CRAFT_RUN')
    else:path.write_text(json.dumps(value,indent=2))
    return value
def verify():
    value=json.loads((REPORTS/'release-lock.json').read_text())
    if value['sources']!=sources() or value['environment']!=environment() or value['gpu']!=GPU or value['threads']!=THREADS:
        raise RuntimeError('Frozen runtime mismatch; do not overwrite a release lock')
    return value

def gpu_lock():
    import fcntl
    folder=DATA/'.locks';folder.mkdir(parents=True,exist_ok=True)
    handle=(folder/(hashlib.sha256(GPU.encode()).hexdigest()+'.lock')).open('a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return handle
