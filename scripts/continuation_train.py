"""Experimental continuation entry; not approved for formal runs before GPU parity."""
import os,sys,resource,argparse,json,time,pickle,hashlib,fcntl,faulthandler
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'dreamerv3'))
import project_runtime as rt
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
rt.affinity()
import jax
import numpy as np
import elements,embodied
import ruamel.yaml as yaml
from embodied.jax import internal
from dreamerv3 import main as upstream
from continuation_adapter import ContinuationCrafter
from exact_replay import ExactReplay
from training_runtime import TrainingRuntime

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trace',action='store_true')
    ap.add_argument('--protocol',required=True)
    ap.add_argument('--deadline',type=float,required=True)
    ap.add_argument('--experimental',action='store_true')
    ap.add_argument('--target',type=int,required=True)
    ap.add_argument('--logdir',required=True)
    args=ap.parse_args()
    if not args.experimental:raise ValueError('GPU validation pending; explicit experimental opt-in required')
    protocol_path=Path(args.protocol).resolve()
    protocol=json.loads(protocol_path.read_text())
    if not protocol['source_step'] < args.target <= protocol['target']:raise ValueError('Invalid segment target')
    if args.deadline<=time.time()+60:raise ValueError('Deadline leaves no useful time')
    original=Path(protocol['source_checkpoint']).resolve()
    for filename in ['agent.pkl','replay.pkl','runtime.pkl','step.pkl','done']:
        if not (original/filename).is_file():raise ValueError('Source checkpoint incomplete')
    assert int(pickle.loads((original/'step.pkl').read_bytes()))==protocol['source_step']
    assert hashlib.sha256((original/'agent.pkl').read_bytes()).hexdigest()==protocol['source_agent_sha256']
    rt.freeze()
    gpu_lock=rt.gpu_lock()
    log=rt.owned(args.logdir)
    assert log.is_relative_to(rt.MODELS)
    log.mkdir(parents=True,exist_ok=True)
    lock=(log/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    saved_protocol=log/'protocol.json'
    if saved_protocol.exists() and json.loads(saved_protocol.read_text())!=protocol:
        raise ValueError('Protocol changed on resume')
    saved_protocol.write_text(json.dumps(protocol,indent=2))
    config_path=Path(protocol['source_config']).resolve()
    assert hashlib.sha256(config_path.read_bytes()).hexdigest()==protocol['source_config_sha256']
    cfg=yaml.YAML(typ='safe').load(config_path.read_text())
    cfg['jax']['profiler']=False
    config=elements.Config(cfg).update(seed=protocol["seed"],logdir=str(log))
    config.save(elements.Path(str(log/'config.yaml')))
    env=ContinuationCrafter(protocol)
    upstream.make_env=lambda config,index:ContinuationCrafter(protocol)
    replay_kind=protocol.get("replay_kind","uniform")
    if replay_kind not in ("uniform","curiosity"):
        raise ValueError("Unknown replay kind")
    replay_kwargs=dict(length=config.consec_train*config.batch_length+config.replay_context,
        capacity=int(config.replay.size),chunksize=config.replay.chunksize,seed=protocol["seed"])
    if replay_kind=="curiosity":
        from curiosity_agent import CuriousAgent
        from curiosity_replay import CuriosityReplay
        from dreamerv3 import agent as agent_module
        replay=CuriosityReplay(**replay_kwargs,curiosity=protocol["curiosity"])
        original_agent=agent_module.Agent
        try:
            agent_module.Agent=CuriousAgent
            agent=upstream.make_agent(config)
        finally:
            agent_module.Agent=original_agent
    else:
        if "curiosity" in protocol:
            raise ValueError("Curiosity settings supplied for uniform replay")
        agent=upstream.make_agent(config)
        replay=ExactReplay(**replay_kwargs)
    stream=iter(upstream.make_stream(config,replay,'train'))
    driver=embodied.Driver([lambda:env],parallel=False)
    driver.reset(agent.init_policy)
    carry=[agent.init_train(config.batch_size)]
    ratio=elements.when.Ratio(config.run.train_ratio/(config.batch_size*config.batch_length))
    runtime=TrainingRuntime(agent,driver,env,stream,ratio,carry)
    counter=elements.Counter()
    cp=elements.Checkpoint(log/'ckpt',keep=None)
    cp.step=counter;cp.agent=agent;cp.replay=replay;cp.runtime=runtime
    if cp.latest():
        cp.load()
    else:
        migration_report={}
        if replay_kind=="curiosity":
            if protocol.get("source_replay_kind")!="uniform":
                raise ValueError("Initial CR migration requires explicit uniform source")
            class InitialReplayMigration:
                def load(self,data):
                    migration_report.update(replay.migrate_uniform(data))
            cp.replay=InitialReplayMigration()
        try:
            cp.load(str(original))
        finally:
            cp.replay=replay
        if migration_report:
            (log/'replay-migration.json').write_text(json.dumps(migration_report,indent=2))
        assert int(counter)==protocol['source_step'] and len(replay)==int(config.replay.size) and env._env.done
        (log/'initialization.json').write_text(json.dumps(dict(source=str(original),step=int(counter),replay_size=len(replay),updates=int(agent.n_updates),mode='full replay/runtime/optimizer restore')))

    assert int(counter)==env._env.actions and env._env.settings['target']==protocol['target']
    # WAL records physical execution separately from retained training steps.
    physical=(log/'physical-actions.jsonl').open('a',buffering=1)
    original_step=env._env.step
    def measured_step(action):
        record=dict(pid=os.getpid(),retained_index=env._env.actions+1,
                    assisted=env._env.last_reset_audit['assisted'])
        physical.write(json.dumps(dict(event='begin',**record))+'\n')
        physical.flush();os.fsync(physical.fileno())
        before=dict(env._env.env._player.achievements)
        result=original_step(action)
        unlocks=[k for k,v in env._env.env._player.achievements.items() if v>0 and before[k]==0]
        physical.write(json.dumps(dict(event='end',new_unlocks=unlocks,skill=env._env.kind,**record))+'\n')
        physical.flush();os.fsync(physical.fileno())
        return result
    env._env.step=measured_step
    started=time.time()
    status=dict(state='running',pid=os.getpid(),start_actions=env._env.actions,target=args.target,started=started)
    def save_status():
        status.update(actions=env._env.actions,assisted_actions=env._env.assisted_actions,
                      updates=int(agent.n_updates),heartbeat=time.time(),counts=env._env.counts,preparation_actions=env._env.preparation_actions,random_overrides=env._env.exploration.overrides)
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
        if args.trace: trace('transition',data=fingerprint({k:v for k,v in tran.items() if not k.startswith('log/')}))
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
                if args.trace: trace('batch',fields={k:fingerprint(v) for k,v in batch.items() if k!='stepid'},batch=fingerprint({k:v for k,v in batch.items() if k!='stepid'}),carry=fingerprint(carry[0]),batchid=int(agent.n_batches))
                batch=internal.device_put(batch,agent.train_sharded)
                with agent.n_batches.lock:
                    batchid=agent.n_batches.value;agent.n_batches.value+=1
                batch['seed']=agent._seeds(batchid,agent.train_mirrored)
                carry[0],outs,mets=agent.train(carry[0],batch)
                if 'replay' in outs:
                    if args.trace: trace('replay_update',fields={k:fingerprint(v) for k,v in outs['replay'].items()})
                    replay.update(outs['replay'])
                assert all(np.isfinite(np.asarray(v)).all() for k,v in mets.items() if 'loss' in k)
        if time.time()-lastlog[0]>30:
            metrics.write(json.dumps(dict(step=int(counter),updates=int(agent.n_updates),
                losses={k:float(np.asarray(v).mean()) for k,v in mets.items() if 'loss' in k}))+'\n')
            save_status();lastlog[0]=time.time()
    driver.on_step(callback)
    def policy(*a):
        pc,act,out=agent.policy(*a,mode='train')
        assisted=env._env.kind not in (None,'natural')
        actual,override=env._env.exploration.choose(int(np.asarray(act['action'])[0]),
            env._env.episode_steps,is_last=bool(np.any(a[1]['is_last'])) or not assisted)
        if override:
            act=dict(act);act['action']=np.array([actual],np.int32)
            with jax.transfer_guard('allow'):
                pc=list(pc);pc[3]={'action':[jax.numpy.array(actual,jax.numpy.int32)]};pc=tuple(pc)
        return pc,act,out
    lastsave=time.time()
    while env._env.actions<args.target and time.time()<args.deadline-30:
        trace('before',carry=fingerprint(driver.carry),policy=fingerprint(agent.policy_params),pending=fingerprint(agent.pending_sync),actions=fingerprint(driver.acts),counter=int(agent.n_actions)) if args.trace else None
        faulthandler.dump_traceback_later(240,repeat=False)
        driver(policy,steps=1)
        faulthandler.cancel_dump_traceback_later()
        if args.trace and env._env.actions==300:cp.save()
        if time.time()-lastsave>=120: # Retain all until evaluated and explicitly audited.
            cp.save();lastsave=time.time()
    cp.save()
    status.update(state='completed' if env._env.actions==args.target else 'budget_stopped',finished=time.time());save_status()
    metrics.close();ledger.close();driver.close()
if __name__=='__main__':
    try:main()
    except BaseException as error:
        if '--logdir' in sys.argv:
            folder=Path(sys.argv[sys.argv.index('--logdir')+1]).resolve()
            if folder.is_relative_to(rt.MODELS):
                folder.mkdir(parents=True,exist_ok=True)
                (folder/'last-error.json').write_text(json.dumps(dict(error=repr(error),time=time.time(),pid=os.getpid())))
        raise
