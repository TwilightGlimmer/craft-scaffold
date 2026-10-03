"""Explicit target-host validation. CPU mode never initializes JAX/CUDA."""
import sys,json,subprocess,argparse,fcntl,hashlib,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import project_runtime as rt
p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');a=p.parse_args()
rt.initialize()
lock=(rt.REPORTS/'formal-worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
subprocess.run([sys.executable,str(rt.ROOT/'scripts/test_curriculum_env.py')],check=True)
subprocess.run([sys.executable,'-m','unittest','discover','-s',str(rt.ROOT/'tests'),'-v'],check=True,cwd=rt.ROOT)
if not a.gpu:
    print('CPU validation passed. GPU/resume validation still required for matrix runs.')
    sys.exit(0)
# The doctor checks device availability only after the explicit GPU flag.
gpu_lock=rt.gpu_lock()
from resource_guard import run,gpu_query
if gpu_query('pid',True):raise RuntimeError('GPU occupied')
subprocess.run([sys.executable,str(rt.ROOT/'tools/doctor.py'),'--gpu'],check=True)
rt.freeze()
gate=rt.REPORTS/'local-validation.json'
if gate.exists():raise RuntimeError('Validation already recorded; use a new CRAFT_RUN for a fresh attempt')
def train(name,target,method='natural'):
    model=rt.MODELS/name
    cmd=[str(rt.ROOT/'python.sh'),str(rt.ROOT/'scripts/curriculum_train.py'),'--method',method,'--seed','7','--target',str(target),'--logdir',str(model)]
    result=run(cmd,rt.LOGS/f'validation-{name}-{target}.log',model/'physical-actions.jsonl')
    if result['exit_code']!=0:raise RuntimeError(result)
    return model
def latest(model):
    folders=sorted(x for x in (model/'ckpt').iterdir() if x.is_dir() and (x/'done').exists())
    cp=folders[-1]
    assert all((cp/(name+'.pkl')).exists() for name in ['step','agent','replay','runtime'])
    return cp
# Fresh paths only: never mistake existing checkpoints for a completed new validation.
for name in ['validation-continuous','validation-resumed','validation-boundary']:
    if (rt.MODELS/name).exists():raise RuntimeError('Incomplete validation exists; choose a new CRAFT_RUN')
continuous=train('validation-continuous',400)
for target in [300,302,400]:resumed=train('validation-resumed',target)
# Compare model arrays recursively without initializing another JAX device.
import numpy as np
def compare(x,y):
    if isinstance(x,dict):
        assert x.keys()==y.keys()
        return sum(compare(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):
        assert type(x)==type(y) and len(x)==len(y)
        return sum(compare(a,b) for a,b in zip(x,y))
    if isinstance(x,np.ndarray):
        assert isinstance(y,np.ndarray) and x.dtype==y.dtype and np.array_equal(x,y)
        return 1
    assert x==y
    return 0
c1,c2=latest(continuous),latest(resumed)
arrays=compare(pickle.loads((c1/'agent.pkl').read_bytes()),pickle.loads((c2/'agent.pkl').read_bytes()))
assert arrays>0
boundary=train('validation-boundary',5100,'fixed_mix')
s=json.loads((boundary/'status.json').read_text())
assert s['actions']==5100 and s['assisted_actions']==2500
dest=rt.REPORTS/'validation-evaluation'
result=run([str(rt.ROOT/'python.sh'),str(rt.ROOT/'scripts/curriculum_evaluate.py'),'--model',str(boundary),'--output',str(dest),'--step','5100','--development'],rt.LOGS/'validation-evaluation.log',dest/'status.json')
if result['exit_code']!=0:raise RuntimeError(result)
subprocess.run([sys.executable,str(rt.ROOT/'scripts/analyze_curriculum_evaluation.py'),'--directory',str(dest)],check=True)
assert json.loads((dest/'diagnostic-gate.json').read_text())['passed']
rt.verify()
gate.write_text(json.dumps(dict(passed=True,arrays_compared=arrays,source_hashes=rt.sources(),
    environment=rt.environment(),scope='Target-host 400-step resume,5100-step boundary,100 development episodes; not formal performance'),indent=2))
print('Local GPU validation passed. Matrix run may now start.')
