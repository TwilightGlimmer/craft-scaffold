"""Experimental native evaluation; auxiliary and natural groups stay separate."""
import os
import os,sys,json,time,argparse,hashlib,pickle
from pathlib import Path
import numpy as np
from crafter import constants
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'dreamerv3'))
import project_runtime as rt
rt.affinity()
from skill_starts import StableCrafterEnv,prepare_native as prepare,SKILLS,TARGETS
SCHEMA='native-continuation-evaluation-v1'
def achievement_delta(initial,current):
 return {k:int(initial[k]==0 and current[k]>0) for k in initial}
LEVELS=SKILLS
import elements,ruamel.yaml
from dreamerv3.main import make_agent
ap=argparse.ArgumentParser()
for k in ('checkpoint','config','output','kind'):ap.add_argument('--'+k,required=True)
ap.add_argument('--policy-seed',type=int,default=9203);ap.add_argument('--episodes',type=int,default=30);ap.add_argument('--namespace',type=int,default=9201)
ap.add_argument('--experimental',action='store_true')
a=ap.parse_args()
if not a.experimental:raise ValueError('Pending GPU parity; explicit experimental opt-in required')
if a.episodes<=0:raise ValueError('Episodes must be positive')
out=rt.owned(a.output)
rt.freeze()
gpu_lock=rt.gpu_lock()
out.mkdir(parents=True,exist_ok=False)
def write(name,v):
 t=out/(name+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(out/(name+'.json'))
config=elements.Config(ruamel.yaml.YAML(typ='safe').load(Path(a.config).read_text())).update(seed=a.policy_seed,logdir=str(out))
cpdir=Path(a.checkpoint)
assert all((cpdir/n).is_file() for n in ['agent.pkl','step.pkl','done'])
sha=hashlib.sha256((cpdir/'agent.pkl').read_bytes()).hexdigest()
agent=make_agent(config);cp=elements.Checkpoint();cp.agent=agent;cp.load(str(cpdir),keys=['agent'])
conditions=list(LEVELS) if a.kind=='all' else a.kind.split(',')
if not conditions or len(set(conditions))!=len(conditions) or any(k not in ('natural',)+LEVELS for k in conditions):raise ValueError(a.kind)
records=[];started=time.time()
trace=(out/'actions.jsonl').open('w',buffering=1)
def plain(v):
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,np.ndarray):return v.tolist()
 raise TypeError(type(v).__name__)

for level in conditions:
 for ep in range(a.episodes):
  seed=int(np.random.SeedSequence([a.namespace,ep]).generate_state(1)[0])
  env=StableCrafterEnv(seed=seed);obs=env.reset();obs=prepare(env,level,ep%4)
  initial=dict(env._player.achievements);reset_audit=env._auxiliary_audit
  carry=agent.init_policy(1)
  with agent.n_actions.lock:agent.n_actions.value=ep*10001
  reward=0.;total=0.;native_total=0.;suppressed=0.;steps=0;done=False
  first_success=None;wasted_woodpick=0;placed_stone=0
  while not done:
   legal=dict(image=obs[None],reward=np.array([reward],np.float32),is_first=np.array([steps==0]),is_last=np.array([False]),is_terminal=np.array([False]))
   carry,act,_=agent.policy(carry,legal,mode='eval')
   before_inv=dict(env._player.inventory)
   before_ach=dict(env._player.achievements)
   nearby,_=env._player.world.nearby(env._player.pos,1)
   recipe=constants.make.get({'woodpick':'wood_pickaxe','stonepick':'stone_pickaxe','ironpick':'iron_pickaxe'}.get(level,''))
   missing={k:int(v-before_inv[k]) for k,v in (recipe['uses'].items() if recipe else []) if before_inv[k]<v}
   missing_util=[k for k in (recipe['nearby'] if recipe else []) if k not in nearby]
   audit=dict(level=level,episode=ep,step=steps+1,environment_seed=seed,action=constants.actions[int(act['action'][0])],
      position=list(env._player.pos),inventory_before=before_inv,
      missing_materials=missing,missing_utilities=missing_util,
      recipe_ready=(not missing and not missing_util) if recipe else None,
      sleeping=bool(getattr(env._player,'sleeping',False)))

   obs,reward,done,info=env.step(int(act['action'][0]));total+=reward;native_total+=reward;steps+=1
   audit.update(inventory_after=dict(env._player.inventory),reward=float(reward),
       native_reward=float(reward),done=bool(done),
       achievement_increments={k:int(v-before_ach[k]) for k,v in info['achievements'].items() if v!=before_ach[k]})
   trace.write(json.dumps(audit,default=plain)+'\n')
   if first_success is None:
    wasted_woodpick+=int(before_inv['wood_pickaxe']>0 and env._player.inventory['wood_pickaxe']>before_inv['wood_pickaxe'])
    placed_stone+=int(constants.actions[int(act['action'][0])]=='place_stone' and env._player.inventory['stone']<before_inv['stone'])
    if level!='natural' and initial[TARGETS[level]]==0 and info['achievements'][TARGETS[level]]>0:first_success=steps
   if steps%100==0:write('status',dict(state='running',episodes=len(records),step=steps,heartbeat=time.time()))
  records.append(dict(first_target_step=first_success,repeat_woodpick_before_success=wasted_woodpick,place_stone_before_success=placed_stone,level=level,episode=ep,environment_seed=seed,steps=steps,return_=total,native_return=native_total,suppressed_achievement_bonus=suppressed,achievements=achievement_delta(initial,info['achievements']),achievement_counter_deltas={k:int(info['achievements'][k])-int(initial[k]) for k in initial},initial_achievements=initial,final_achievements=info['achievements'],reset_audit=reset_audit))
  write('episodes',records);write('status',dict(state='running',episodes=len(records),heartbeat=time.time()))
  env.close()
trace.close()
groups={}
for level in conditions:
 rows=[r for r in records if r['level']==level]
 rates={k:100*sum(r['achievements'][k]>0 for r in rows)/len(rows) for k in rows[0]['achievements']}
 groups[level]=dict(episodes=len(rows),target_successes=sum(r['achievements'][TARGETS[level]]>0 for r in rows) if level!='natural' else None,score=float(np.exp(np.mean(np.log1p(list(rates.values()))))-1),achievement_rates=rates,mean_return=float(np.mean([r['return_'] for r in rows])),mean_steps=float(np.mean([r['steps'] for r in rows])))
assert hashlib.sha256((cpdir/'agent.pkl').read_bytes()).hexdigest()==sha
write('summary',dict(schema=SCHEMA,checkpoint_step=int(pickle.loads((cpdir/'step.pkl').read_bytes())),checkpoint=str(cpdir),checkpoint_sha256=sha,namespace=a.namespace,policy_seed=a.policy_seed,groups=groups,episodes=len(records),elapsed=time.time()-started))
write('status',dict(state='completed',episodes=len(records),heartbeat=time.time()))
