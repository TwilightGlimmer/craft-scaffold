"""Sequential three-method three-seed runner; frozen release required."""
import json,time,subprocess,psutil,fcntl,pickle,hashlib
import project_runtime as rt
from resource_guard import run as guarded_run
rt.initialize()
gpu_lock=rt.gpu_lock()
ROOT,OUT,BASE,GPU=rt.ROOT,rt.REPORTS,rt.MODELS,rt.GPU
lock=(OUT/'formal-worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=dict(state='starting',pid=os.getpid(),created=psutil.Process().create_time(),started=time.time())
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def status():write(OUT/'formal-status.json',state)
def append(path,obj):
    with path.open('a') as f:f.write(json.dumps(obj)+'\n')
def check_sources():rt.verify()
def checkpoint(model):
    root=model/'ckpt'
    if not root.exists():return 0,None
    for cp in sorted((x for x in root.iterdir() if x.is_dir()),reverse=True):
        if (cp/'done').exists() and all((cp/f'{x}.pkl').exists() for x in ['agent','replay','runtime','step']):
            step=int(pickle.loads((cp/'step.pkl').read_bytes()))
            (root/'latest').write_text(cp.name)
            return step,cp
    return 0,None
def run(cmd,tag,progress):
    check_sources()
    def update(**values):
        state.update(values);status()
    return guarded_run(cmd,rt.LOGS/f'{tag}-{int(time.time())}.log',progress,update)
try:
    gate=json.loads((OUT/'local-validation.json').read_text())
    assert gate['passed'] and gate['source_hashes']==rt.sources() and gate['environment']==rt.environment()
    protocol=json.loads((ROOT/'configs/protocol.json').read_text())
    check_sources();status()
    matrix=[(method,seed) for seed in [11,23,41] for method in ['natural','fixed_mix','fading_mix']]
    models=[]
    for method,seed in matrix:
        model=BASE/f'{method}-seed{seed}';models.append((method,seed,model))
        step,cp=checkpoint(model);failures=0
        while step<300000:
            target=min(step+10000,300000)
            state.update(state='training',method=method,seed=seed,checkpoint_step=step,target=target);status()
            result=run([str(ROOT/'python.sh'),str(ROOT/'scripts/curriculum_train.py'),'--method',method,'--seed',str(seed),'--target',str(target),'--logdir',str(model)],f'{method}-{seed}-{target}',model/'physical-actions.jsonl')
            previous=step;step,cp=checkpoint(model)
            append(OUT/'formal-attempts.jsonl',dict(method=method,seed=seed,start_step=previous,checkpoint_step=step,target=target,**result))
            failures=failures+1 if step<=previous else 0
            assert result['reason']!='resource_limit','Resource limit requires diagnosis'
            assert failures<5,'Five no-progress attempts require diagnosis'
        s=json.loads((model/'status.json').read_text())
        assert s['actions']==300000 and s['assisted_actions']==(0 if method=='natural' else 75000)
        state.update(checkpoint_step=step);status()
    # Freeze every final weight before any formal evaluation.
    final={}
    for method,seed,model in models:
        step,cp=checkpoint(model);assert step==300000
        final[str(model)]=dict(checkpoint=str(cp),agent_sha256=hashlib.sha256((cp/'agent.pkl').read_bytes()).hexdigest())
    write(OUT/'final-checkpoints.json',final);check_sources()
    for method,seed,model in models:
        evalroot=OUT/'formal-evaluations';evalroot.mkdir(exist_ok=True)
        for attempt in range(1,6):
            dest=evalroot/f'{method}-seed{seed}-attempt{attempt}'
            if (dest/'diagnostic-gate.json').exists() and json.loads((dest/'diagnostic-gate.json').read_text())['passed']:break
            if dest.exists():continue
            state.update(state='evaluating',method=method,seed=seed);status()
            result=run([str(ROOT/'python.sh'),str(ROOT/'scripts/curriculum_evaluate.py'),'--model',str(model),'--output',str(dest),'--step','300000'],f'eval-{method}-{seed}-{attempt}',dest/'status.json')
            append(OUT/'formal-evaluation-attempts.jsonl',dict(method=method,seed=seed,attempt=attempt,**result))
            if result['exit_code']==0:
                subprocess.run([str(ROOT/'python.sh'),str(ROOT/'scripts/analyze_curriculum_evaluation.py'),'--directory',str(dest)],check=True)
                assert json.loads((dest/'summary.json').read_text())['checkpoint_sha256']==final[str(model)]['agent_sha256']
                assert json.loads((dest/'diagnostic-gate.json').read_text())['passed']
                break
        else:raise RuntimeError('Five incomplete evaluations')
    check_sources();state.update(state='completed',finished=time.time());status()
except BaseException as e:
    state.update(state='failed',error=repr(e),finished=time.time());status();raise
