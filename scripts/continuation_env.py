"""Native-reward continuation environment with explicit immutable run settings."""
import pickle
import numpy as np
from skill_starts import StableCrafterEnv,prepare_native,KINDS
from legacy_native_state import load_legacy_native_state

class ContinuationEnv:
    def __init__(self,seed,target,weights,namespace,block_actions=2500,
                 auxiliary_horizon=128,**kwargs):
        if set(weights)-set(KINDS):raise ValueError("Unknown skill in weights")
        weights={k:float(weights.get(k,0)) for k in KINDS}
        if any(v<0 or not np.isfinite(v) for v in weights.values()) or not np.isclose(sum(weights.values()),1):
            raise ValueError("Weights must be finite, nonnegative and sum to one")
        if target<=0 or block_actions<=0 or auxiliary_horizon<=0:
            raise ValueError("Budgets must be positive")
        self.settings=dict(seed=int(seed),target=int(target),weights=weights,
                          namespace=int(namespace),block_actions=int(block_actions),
                          auxiliary_horizon=int(auxiliary_horizon),kwargs=kwargs)
        self.env=StableCrafterEnv(seed=seed,**kwargs)
        self.observation_space=self.env.observation_space
        self.action_space=self.env.action_space
        self.actions=self.origin=self.resets=self.preparation_actions=0
        self.assisted_actions=0
        self.counts=dict.fromkeys(KINDS,0)
        self.done=True;self.kind=None;self.block=-1;self.episode_steps=0
        self.last_reset_audit={}

    def migrate_legacy(self,data,expected_step):
        """Use only at a completed episode; agent/replay/runtime restore is separate."""
        if self.actions or not self.done:raise ValueError("Migration only allowed before continuation")
        state=load_legacy_native_state(data)
        if state["actions"]!=expected_step or not state["done"]:
            raise ValueError("Migration requires the selected completed-episode checkpoint")
        if state["seed"]!=self.settings["seed"] or expected_step>=self.settings["target"]:
            raise ValueError("Source seed or target incompatible")
        self.env.close();self.env=state["env"]
        self.actions=self.origin=int(expected_step)
        self.resets=int(state["resets"])
        self.assisted_actions=int(state["assisted_actions"])
        self.last_reset_audit=state["last_reset_audit"]
        self.done=True

    def reset(self):
        if not self.done:raise RuntimeError("Active episode cannot be discarded")
        if self.actions>=self.settings["target"]:raise RuntimeError("Training budget exhausted")
        size=self.settings["block_actions"]
        block=(self.actions-self.origin)//size
        if block!=self.block:
            n=sum(self.counts.values())+size
            eligible=[k for k in KINDS if self.settings["weights"][k]>0]
            self.kind=max(eligible,key=lambda k:self.settings["weights"][k]*n-self.counts[k])
            self.block=block
        seed=int(np.random.SeedSequence([self.settings["namespace"],self.settings["seed"],self.resets]).generate_state(1)[0]%(2**31-1))
        self.env.close()
        self.env=StableCrafterEnv(seed=seed,**self.settings["kwargs"])
        self.env.reset()
        image=prepare_native(self.env,self.kind,self.resets%4)
        audit=self.env._auxiliary_audit
        self.preparation_actions+=audit["preparation_actions"]
        self.last_reset_audit=dict(seed=seed,skill=self.kind,assisted=self.kind!="natural",
                                   actions=self.actions,reset_audit=audit)
        self.resets+=1;self.episode_steps=0;self.done=False
        return image

    def step(self,action):
        if self.done:raise RuntimeError("Reset required")
        if self.actions>=self.settings["target"]:raise RuntimeError("Training budget exhausted")
        obs,reward,done,info=self.env.step(action)
        self.actions+=1;self.episode_steps+=1;self.counts[self.kind]+=1
        self.assisted_actions+=int(self.kind!="natural")
        cutoff=((self.actions-self.origin)%self.settings["block_actions"]==0
                or self.actions==self.settings["target"]
                or (self.kind!="natural" and self.episode_steps>=self.settings["auxiliary_horizon"]))
        if cutoff and not done:done=True;info=dict(info,discount=1.)
        self.done=bool(done)
        return obs,reward,self.done,info

    def save(self):
        state=self.__dict__.copy()
        state.pop("step",None)
        return pickle.dumps(state,pickle.HIGHEST_PROTOCOL)

    def load(self,data):
        state=pickle.loads(data)
        if state["settings"]!=self.settings:raise ValueError("Continuation protocol changed")
        self.env.close()
        self.__dict__.update(state)

    def close(self):self.env.close()
