
from pathlib import Path
import argparse,json,collections,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True);args=ap.parse_args()
import project_runtime as rt
OUT=rt.owned(args.directory)
status=json.loads((OUT/'status.json').read_text())
assert status['state']=='completed' and status['episodes_completed']==100
records=json.loads((OUT/'episodes.json').read_text())
assert len(records)==100
actions=collections.Counter();eligible_actions=collections.Counter();fail=collections.Counter();lost=collections.Counter()
episode_opportunities=[];attempt_records=[];table_without_wood=0;table_total=0;total=0;successes=0;after_first_success=0;pre_first_opportunity_steps=0
for ep,rec in enumerate(records):
 rows=[json.loads(x) for x in (OUT/f'episode-{ep:03}.jsonl').read_text().splitlines()]
 assert len(rows)==rec['steps'] and rows[-1]['done']
 elig=attempt=valid=made=0
 for t,row in enumerate(rows):
  assert row['step']==t and row['episode']==ep and row['environment_seed']==rec['environment_seed']
  before=row['before'];after=row['after'];name=row['action_name']
  assert before['eligible']==(before['materials_ok'] and before['near_table'] and not before['action_overridden'])
  assert row['wood_pickaxe_made']==(after['achievements']['make_wood_pickaxe']>before['achievements']['make_wood_pickaxe'])
  if t:assert before==rows[t-1]['after']
  elig+=int(before['eligible']);attempt+=int(name=='make_wood_pickaxe');valid+=int(name=='make_wood_pickaxe' and before['eligible']);made+=int(row['wood_pickaxe_made'])
  total+=1;actions[name]+=1
  if before['eligible']:eligible_actions[name]+=1
  if before['achievements']['make_wood_pickaxe']==0:pre_first_opportunity_steps+=int(before['eligible'])
  else:after_first_success+=int(before['eligible'])
  fail.update(row['attempt_failure_reasons']);lost.update(row['opportunity_loss_reasons'])
  if name=='place_table' and after['achievements']['place_table']>before['achievements']['place_table']:
   table_total+=1;table_without_wood+=int(after['inventory']['wood']<1)
  if name=='make_wood_pickaxe':attempt_records.append(dict(episode=ep,step=t,eligible=before['eligible'],made=row['wood_pickaxe_made'],reasons=row['attempt_failure_reasons']))
 assert (elig,attempt,valid,made)==(rec['eligible_steps'],rec['make_attempts'],rec['eligible_attempts'],rec['wood_pickaxes_made'])
 assert np.isclose(sum(r['reward'] for r in rows),rec['return_'])
 assert rec['distinct']==sum(v>0 for v in rows[-1]['after']['achievements'].values())
 successes+=int(made>0)
 if elig:episode_opportunities.append(dict(episode=ep,steps=elig,longest=rec['longest_eligible_streak'],attempts=valid,success=bool(made)))
n=len(episode_opportunities)
result=dict(episodes=100,total_steps=total,episodes_with_opportunity=n,opportunity_episode_rate=n/100,total_opportunity_steps=sum(r['eligible_steps'] for r in records),pre_first_success_opportunity_steps=pre_first_opportunity_steps,post_success_opportunity_steps=after_first_success,successful_episodes=successes,success_given_opportunity=successes/n if n else None,eligible_action_counts=dict(eligible_actions),all_action_counts=dict(actions),attempt_failures=dict(fail),opportunity_losses=dict(lost),tables_placed=table_total,tables_leaving_no_wood=table_without_wood,opportunity_episodes=episode_opportunities,make_attempts=attempt_records,interpretation='Descriptive diagnostics only. A low opportunity count cannot establish a policy-choice failure; success conditioned on opportunity is not a causal effect.')
(OUT/'diagnostic-analysis.json').write_text(json.dumps(result,indent=2))
unexpected=fail.get('unexpected_failure',0)
gate=dict(passed=unexpected==0,trace_integrity=True,episodes=100,unexpected_eligible_craft_failures=unexpected,scope='Instrumentation and trajectory audit; does not certify model quality or select an adaptation')
(OUT/'diagnostic-gate.json').write_text(json.dumps(gate,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['make_attempts','opportunity_episodes']}))
print('GATE',json.dumps(gate))
