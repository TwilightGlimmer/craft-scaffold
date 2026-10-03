import json,sys
from pathlib import Path
import numpy as np
import crafter
from curriculum_env import CurriculumEnv,assisted_block,StableCrafterEnv
ROOT=Path(__file__).resolve().parents[1]
def equal(a,b):
    assert np.array_equal(a[0],b[0])
    assert a[1:3]==b[1:3]
    for key in ['inventory','achievements','discount','reward']:
        assert a[3][key]==b[3][key],key
    assert np.array_equal(a[3]['semantic'],b[3]['semantic'])
results={}
for method,expected in [('natural',0),('fixed_mix',75000),('fading_mix',75000)]:
    assert sum(assisted_block(method,i)*2500 for i in range(120))==expected
results['exact_schedule_counts']=True
env=CurriculumEnv('natural',7);image=env.reset()
ref=StableCrafterEnv(seed=env.last_reset_audit['seed']);assert np.array_equal(image,ref.reset())
rng=np.random.RandomState(42)
for _ in range(100):
    a=int(rng.randint(17));x=env.step(a);y=ref.step(a);equal(x,y)
    if x[2]:break
results['natural_matches_stable_reference']=True
env=CurriculumEnv('fixed_mix',7);image=env.reset()
assert env.env._player.inventory['wood']==1
assert all(v==0 for v in env.env._player.achievements.values())
assert np.array_equal(image,env.env._obs())
out=env.step(crafter.constants.actions.index('make_wood_pickaxe'))
assert out[3]['achievements']['make_wood_pickaxe']==1
assert out[3]['inventory']['wood_pickaxe']==1
results['assisted_craft_without_granted_achievement']=True
saved=env.save()
clone=CurriculumEnv('fixed_mix',7);clone.load(saved)
for _ in range(100):
    action=int(rng.randint(17));x=env.step(action);y=clone.step(action);equal(x,y)
    assert x[3]['curriculum_audit']==y[3]['curriculum_audit']
    if x[2]:break
results['environment_and_rng_resume_equivalence']=True
env=CurriculumEnv('fixed_mix',7);env.reset()
env.actions=2499
out=env.step(crafter.constants.actions.index('noop'))
assert out[2] and out[3]['discount']==1
assert out[3]['curriculum_audit']['block_boundary']
env.reset();assert not env.last_reset_audit['assisted']
try:
    env.reset()
    raise AssertionError('premature reset accepted')
except RuntimeError:pass
results['boundary_truncation_and_reset_guard']=True
env.env._player.health=0
out=env.step(crafter.constants.actions.index('noop'))
assert out[2] and out[3]['discount']==0
results['death_terminal']=True
result=dict(passed=True,checks=results,scope='CPU environment wrapper only; not yet Dreamer integration or end-to-end checkpoint consistency')
(ROOT/'reports/curriculum-v1/environment-tests.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
