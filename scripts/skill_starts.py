"""Prepared skill starts with native rewards; no action labels enter replay."""
from curriculum_env import StableCrafterEnv
from crafter import constants
SKILLS=('woodpick','stone','coal','stonepick','furnace','iron','ironpick','diamond')
TARGETS=dict(zip(SKILLS,('make_wood_pickaxe','collect_stone','collect_coal','make_stone_pickaxe','place_furnace','collect_iron','make_iron_pickaxe','collect_diamond')))
KINDS=('natural',)+SKILLS
SCHEMA='single-skill-curriculum-v5'
MAINLINE=frozenset(TARGETS.values())|{'collect_wood','place_table'}
def _prepare(env,skill,rotation=0):
    if skill not in KINDS: raise ValueError(skill)
    p,w=env._player,env._world
    assert env._step==0 and not any(p.achievements.values()) and not env._unlocked
    audit=dict(schema=SCHEMA,skill=skill,rotation=rotation,prefix=[],terrain_edits=[],
               source='constructed terrain plus native prerequisite actions; not demonstrations in replay')
    if skill!='natural':
        directions=((1,0),(0,1),(-1,0),(0,-1))
        center=tuple(map(int,p.pos))
        def xy(i,distance=1):
            dx,dy=directions[(rotation+i)%4]
            return (center[0]+distance*dx,center[1]+distance*dy)
        def tile(pos,material):
            before,obj=w[pos]
            if obj is not None: w.remove(obj)
            w[pos]=material
            audit['terrain_edits'].append(dict(pos=pos,before=before,after=material,
                removed=type(obj).__name__ if obj else None))
        def face(i): p.facing=directions[(rotation+i)%4]
        def act(name):
            _,reward,done,_=env.step(constants.actions.index(name))
            assert not done and tuple(p.pos)==center
            audit['prefix'].append(dict(action=name,reward=float(reward)))
        def collect(material,n=1):
            face(0)
            for _ in range(n):
                tile(xy(0),material);act('do')
        # Only local terrain is constructed. Natural mode is untouched.
        for dx in range(-2,3):
            for dy in range(-2,3):
                if dx or dy:tile((center[0]+dx,center[1]+dy),'grass')
        collect('tree',2);face(1);act('place_table')
        collect('tree'); # one wood
        if skill!='woodpick':act('make_wood_pickaxe')
        if skill in ('stonepick','iron','ironpick','diamond'):
            collect('tree');collect('stone')
            if skill!='stonepick':act('make_stone_pickaxe')
        if skill=='furnace':collect('stone',4)
        if skill in ('ironpick','diamond'):
            collect('stone',4);face(2);act('place_furnace')
            collect('coal');collect('iron');collect('tree')
            if skill=='diamond':act('make_iron_pickaxe')
        front={'stone':'stone','coal':'coal','iron':'iron','diamond':'diamond'}.get(skill,'grass')
        tile(xy(0),front)
        # Four finite replenishment blocks; no automatic inventory replenishment.
        for i in range(4):tile(xy(i,2),'stone')
        face(0)
        assert p.achievements[TARGETS[skill]]==0 and TARGETS[skill] not in env._unlocked
        if skill=='furnace':assert p.inventory['stone']==4
        if skill=='stone':assert p.inventory['stone']==0
    audit.update(preparation_actions=len(audit['prefix']),initial_achievements=dict(p.achievements),
                 initial_unlocked=sorted(env._unlocked),initial_inventory=dict(p.inventory),native_step=env._step)
    assert env._unlocked=={k for k,v in p.achievements.items() if v>0}
    env._mainline_filter=(skill!='natural')
    audit.update(mainline_reward_filter=env._mainline_filter,allowed_achievements=sorted(MAINLINE),
                 blocked_bonus_achievements=sorted(set(p.achievements)-MAINLINE))
    env._auxiliary_audit=audit
    return env._obs()


def prepare_native(env, skill, rotation=0):
    """Construct a start; caller counts preparation actions outside agent steps.

    Use a freshly reset StableCrafterEnv. No reward filtering is enabled.
    Returns pixels; audit metadata is for accounting, never policy inputs.
    """
    obs=_prepare(env,skill,rotation)
    env._mainline_filter=False
    audit=env._auxiliary_audit
    audit['mainline_reward_filter']=False
    audit['allowed_achievements']=sorted(env._player.achievements)
    audit['blocked_bonus_achievements']=[]
    return obs
