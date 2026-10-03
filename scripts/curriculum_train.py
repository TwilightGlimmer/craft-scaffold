"""Synchronous curriculum training entry; formal release gated by resume tests."""
import os,sys,resource,argparse,json,time,pickle,hashlib,fcntl,faulthandler
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.deps/jax0438'),str(ROOT/'dreamerv3')]
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
import jax
import numpy as np
import elements,embodied
import ruamel.yaml as yaml
from embodied.jax import internal
from dreamerv3 import main as upstream
from curriculum_adapter import CurriculumCrafter
from exact_replay import ExactReplay
from training_runtime import TrainingRuntime

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trace',action='store_true')
    ap.add_argument('--method',required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--target',type=int,required=True)
    ap.add_argument('--logdir',required=True)
    args=ap.parse_args()
    log=Path(args.logdir).resolve()
    assert log.is_relative_to(Path('/root/gpufree-data/hanzhuo/models/crafter-worldmodel'))
    log.mkdir(parents=True,exist_ok=True)
    lock=(log/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=yaml.YAML(typ='safe').load(Path('/root/gpufree-data/hanzhuo/models/crafter-worldmodel/recovery-v1/config.yaml').read_text())
    cfg['jax']['profiler']=False
    config=elements.Config(cfg).update(seed=args.seed,logdir=str(log))
    config.save(elements.Path(str(log/'config.yaml')))
    env=CurriculumCrafter(args.method,args.seed)
    upstream.make_env=lambda config,index:CurriculumCrafter(args.method,args.seed)
    agent=upstream.make_agent(config)
    replay=ExactReplay(length=config.consec_train*config.batch_length+config.replay_context,
        capacity=int(config.replay.size),chunksize=config.replay.chunksize,seed=args.seed)
    stream=iter(upstream.make_stream(config,replay,'train'))
    driver=embodied.Driver([lambda:env],parallel=False)
    driver.reset(agent.init_policy)
    carry=[agent.init_train(config.batch_size)]
    ratio=elements.when.Ratio(config.run.train_ratio/(config.batch_size*config.batch_length))
    runtime=TrainingRuntime(agent,driver,env,stream,ratio,carry)
    counter=elements.Counter()
    cp=elements.Checkpoint(log/'ckpt',keep=2)
    cp.step=counter;cp.agent=agent;cp.replay=replay;cp.runtime=runtime
    if cp.latest():cp.load()
    assert int(counter)==env._env.actions
    # WAL records physical execution separately from retained training steps.
    physical=(log/'physical-actions.jsonl').open('a',buffering=1)
    original_step=env._env.step
    def measured_step(action):
        record=dict(pid=os.getpid(),retained_index=env._env.actions+1,
                    assisted=env._env.last_reset_audit['assisted'])
        physical.write(json.dumps(dict(event='begin',**record))+'\n')
        physical.flush();os.fsync(physical.fileno())
        result=original_step(action)
        physical.write(json.dumps(dict(event='end',**record))+'\n')
        physical.flush();os.fsync(physical.fileno())
        return result
    env._env.step=measured_step
    started=time.time()
    status=dict(state='running',pid=os.getpid(),start_actions=env._env.actions,target=args.target,started=started)
    def save_status():
        status.update(actions=env._env.actions,assisted_actions=env._env.assisted_actions,
                      updates=int(agent.n_updates),heartbeat=time.time())
        tmp=log/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(log/'status.json')
    save_status()
    metrics=(log/'metrics.jsonl').open('a',buffering=1)
    ledger=(log/'interactions.jsonl').open('a',buffering=1)
    lastlog=[time.time()]
    def fingerprint(tree):
        h=hashlib.sha256()
        for v in jax.tree.leaves(jax.device_get(tree)):
            x=np.asarray(v);h.update(str((x.shape,x.dtype)).encode());h.update(x.tobytes())
        return h.hexdigest()
    tracefile=(log/'trace.jsonl').open('a',buffering=1) if args.trace else None
    def trace(kind, **values):
        if tracefile and 298<=env._env.actions<=302:
            tracefile.write(json.dumps(dict(step=env._env.actions,kind=kind,**values))+'\n')

    def callback(tran,worker):
        trace('transition',data=fingerprint({k:v for k,v in tran.items() if not k.startswith('log/')}))
        replay.add(tran,worker)
        actual=not bool(tran['is_first'])
        if actual:
            counter.increment()
            ledger.write(json.dumps(dict(pid=os.getpid(),action_index=int(counter),
                assisted=env._env.last_reset_audit['assisted']))+'\n')
        runtime.episode_return+=float(tran['reward'])
        runtime.episode_length+=int(actual)
        if tran['is_last']:
            metrics.write(json.dumps(dict(step=int(counter),episode_return=runtime.episode_return,
                episode_length=runtime.episode_length))+'\n')
            runtime.episode_return=0.;runtime.episode_length=0
        mets={}
        if actual and len(replay)>=config.batch_size*config.batch_length:
            for _ in range(ratio(counter)):
                batch=next(stream)
                trace('batch',fields={k:fingerprint(v) for k,v in batch.items() if k!='stepid'},batch=fingerprint({k:v for k,v in batch.items() if k!='stepid'}),carry=fingerprint(carry[0]),batchid=int(agent.n_batches))
                batch=internal.device_put(batch,agent.train_sharded)
                with agent.n_batches.lock:
                    batchid=agent.n_batches.value;agent.n_batches.value+=1
                batch['seed']=agent._seeds(batchid,agent.train_mirrored)
                carry[0],outs,mets=agent.train(carry[0],batch)
                if 'replay' in outs:
                    trace('replay_update',fields={k:fingerprint(v) for k,v in outs['replay'].items()})
                    replay.update(outs['replay'])
                assert all(np.isfinite(np.asarray(v)).all() for k,v in mets.items() if 'loss' in k)
        if time.time()-lastlog[0]>30:
            metrics.write(json.dumps(dict(step=int(counter),updates=int(agent.n_updates),
                losses={k:float(np.asarray(v).mean()) for k,v in mets.items() if 'loss' in k}))+'\n')
            save_status();lastlog[0]=time.time()
    driver.on_step(callback)
    policy=lambda *a:agent.policy(*a,mode='train')
    lastsave=time.time()
    while env._env.actions<args.target:
        trace('before',carry=fingerprint(driver.carry),policy=fingerprint(agent.policy_params),pending=fingerprint(agent.pending_sync),actions=fingerprint(driver.acts),counter=int(agent.n_actions)) if args.trace else None
        faulthandler.dump_traceback_later(240,repeat=False)
        driver(policy,steps=1)
        faulthandler.cancel_dump_traceback_later()
        if args.trace and env._env.actions==300:cp.save()
        if time.time()-lastsave>120:
            cp.save();lastsave=time.time()
    cp.save()
    status.update(state='completed',finished=time.time());save_status()
    metrics.close();ledger.close();driver.close()
if __name__=='__main__':
    try:main()
    except BaseException as error:
        if '--logdir' in sys.argv:
            folder=Path(sys.argv[sys.argv.index('--logdir')+1]).resolve()
            if folder.is_relative_to(Path('/root/gpufree-data/hanzhuo/models/crafter-worldmodel')):
                folder.mkdir(parents=True,exist_ok=True)
                (folder/'last-error.json').write_text(json.dumps(dict(error=repr(error),time=time.time(),pid=os.getpid())))
        raise

