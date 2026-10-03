
import os,sys,json,time,hashlib,pickle,resource,fcntl
from pathlib import Path
import numpy as np,psutil
ROOT=Path(__file__).resolve().parents[1]
import project_runtime as rt
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--output',required=True);ap.add_argument('--step',type=int,required=True);ap.add_argument('--development',action='store_true');args=ap.parse_args()
assert args.development or args.step==300000
NAMESPACE=3003 if args.development else 4003
OUT=rt.owned(args.output);OUT.mkdir(parents=True,exist_ok=True)
MODEL=rt.owned(args.model)
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
rt.affinity()
lock=(OUT/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
status=dict(state='starting',pid=os.getpid(),created=psutil.Process().create_time(),started=time.time())
save(OUT/'status.json',status)
try:
 sys.path.insert(0,str(ROOT/'dreamerv3'))
 import crafter,elements,ruamel.yaml
 from curriculum_env import StableCrafterEnv
 from crafter import constants
 from dreamerv3.main import make_agent
 config=elements.Config(ruamel.yaml.YAML(typ='safe').load((MODEL/'config.yaml').read_text())).update(seed=NAMESPACE)
 cpdir=MODEL/'ckpt'/(MODEL/'ckpt/latest').read_text().strip()
 assert int(pickle.loads((cpdir/'step.pkl').read_bytes()))==args.step
 cp_hash=hashlib.sha256((cpdir/'agent.pkl').read_bytes()).hexdigest()
 agent=make_agent(config)
 cp=elements.Checkpoint();cp.agent=agent;cp.load(str(cpdir),keys=['agent'])
 def state(env):
  player=env._player
  nearby,_=env._world.nearby(player.pos,1)
  recipe=constants.make['wood_pickaxe']
  material_ok=all(player.inventory[k]>=v for k,v in recipe['uses'].items())
  station_ok=all(x in nearby for x in recipe['nearby'])
  action_overridden=player.sleeping and player.inventory['energy']<constants.items['energy']['max']
  return dict(inventory={k:int(v) for k,v in player.inventory.items()},pos=[int(x) for x in player.pos],near_table=bool(station_ok),materials_ok=bool(material_ok),sleeping=bool(player.sleeping),action_overridden=bool(action_overridden),eligible=bool(material_ok and station_ok and not action_overridden),achievements={k:int(v) for k,v in player.achievements.items()})
 records=[]
 for ep in range(100):
  seed=int(np.random.SeedSequence([NAMESPACE,ep]).generate_state(1)[0])
  env=StableCrafterEnv(seed=seed);obs=env.reset();carry=agent.init_policy(1)
  with agent.n_actions.lock:agent.n_actions.value=ep*10001
  reward=0.;steps=0;done=False;total=0.;opportunities=0;attempts=0;valid_attempts=0;success=0;lost={};fail={};streak=0;longest=0
  trace=OUT/f'episode-{ep:03}.jsonl'
  assert not trace.exists(),'Refuse overwrite; interrupted diagnosis requires new output version'
  with trace.open('w') as f:
   while not done:
    before=state(env)
    legal={'image':obs[None],'reward':np.array([reward],np.float32),'is_first':np.array([steps==0]),'is_last':np.array([False]),'is_terminal':np.array([False])}
    assert set(legal)=={'image','reward','is_first','is_last','is_terminal'}
    carry,act,_=agent.policy(carry,legal,mode='eval')
    action=int(act['action'][0]);name=constants.actions[action]
    obs,reward,done,info=env.step(action);after=state(env)
    made=after['achievements']['make_wood_pickaxe']>before['achievements']['make_wood_pickaxe']
    opportunities+=int(before['eligible'])
    streak=streak+1 if before['eligible'] else 0;longest=max(longest,streak)
    attempts+=int(name=='make_wood_pickaxe')
    valid_attempts+=int(name=='make_wood_pickaxe' and before['eligible'])
    success+=int(made)
    failure=[]
    if name=='make_wood_pickaxe' and not made:
     if before['action_overridden']:failure.append('sleep_override')
     if not before['materials_ok']:failure.append('insufficient_wood')
     if not before['near_table']:failure.append('no_nearby_table')
     if not failure:failure.append('unexpected_failure')
     for reason in failure:fail[reason]=fail.get(reason,0)+1
    loss=[]
    if before['eligible'] and not after['eligible']:
     if made:loss.append('successful_craft')
     else:
      if done:loss.append('episode_end')
      if not after['materials_ok']:loss.append('material_consumed')
      if not after['near_table']:loss.append('left_table')
      if after['action_overridden']:loss.append('sleep_override')
     for reason in loss:lost[reason]=lost.get(reason,0)+1
    row=dict(episode=ep,step=steps,environment_seed=seed,action=action,action_name=name,reward=float(reward),done=bool(done),before=before,after=after,wood_pickaxe_made=bool(made),attempt_failure_reasons=failure,opportunity_loss_reasons=loss)
    f.write(json.dumps(row)+'\n')
    total+=reward;steps+=1
  rec=dict(episode=ep,environment_seed=seed,steps=steps,return_=float(total),eligible_steps=opportunities,longest_eligible_streak=longest,make_attempts=attempts,eligible_attempts=valid_attempts,wood_pickaxes_made=success,opportunity_losses=lost,attempt_failures=fail,achievements=info['achievements'],distinct=sum(v>0 for v in info['achievements'].values()))
  records.append(rec)
  save(OUT/'episodes.json',records)
  status.update(state='running',episodes_completed=len(records),heartbeat=time.time());save(OUT/'status.json',status)
 assert hashlib.sha256((cpdir/'agent.pkl').read_bytes()).hexdigest()==cp_hash
 counts={k:sum(r[k]>0 for r in records) for k in ['eligible_steps','make_attempts','eligible_attempts','wood_pickaxes_made']}
 summary=dict(checkpoint=str(cpdir),checkpoint_sha256=cp_hash,episodes=100,episode_counts=counts,total_eligible_steps=sum(r['eligible_steps'] for r in records),total_make_attempts=sum(r['make_attempts'] for r in records),total_eligible_attempts=sum(r['eligible_attempts'] for r in records),mean_distinct=float(np.mean([r['distinct'] for r in records])),mean_return=float(np.mean([r['return_'] for r in records])),scope='Original natural scenarios with stable object ordering;100 fixed episodes; metadata logging only; frozen final checkpoint; explicit evaluation RNG reset',namespace=NAMESPACE,elapsed_seconds=time.time()-status['started'])
 save(OUT/'summary.json',summary)
 status.update(state='completed',finished=time.time(),episodes_completed=100);save(OUT/'status.json',status)
 print(json.dumps(summary),flush=True)
except BaseException as e:
 status.update(state='failed',error=repr(e),finished=time.time());save(OUT/'status.json',status);raise
