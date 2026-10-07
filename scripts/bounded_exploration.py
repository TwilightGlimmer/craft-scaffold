"""Bounded behavior exploration, independent of rewards and task labels."""
import copy
import numpy as np

class BoundedExploration:
    def __init__(self, seed, enabled, action_count=17, early_steps=32, probability=0.5):
        if action_count <= 0 or early_steps < 0 or not 0 <= probability <= 1:
            raise ValueError("Invalid exploration settings")
        self.enabled=bool(enabled)
        self.action_count=int(action_count)
        self.early_steps=int(early_steps)
        self.probability=float(probability)
        self.rng=np.random.default_rng(seed)
        self.overrides=0

    def choose(self, actor_action, episode_steps, is_last=False):
        """Draw in both arms, even if disabled. Return actual action and override flag.

        Caller must write the actual action to replay AND previous-action carry.
        This class cannot perform that architecture-specific carry update.
        """
        coin=float(self.rng.random())
        candidate=int(self.rng.integers(self.action_count))
        replace=(self.enabled and not is_last and episode_steps < self.early_steps
                 and coin < self.probability)
        self.overrides+=int(replace)
        return (candidate if replace else int(actor_action)),bool(replace)

    def state_dict(self):
        return dict(version=1,enabled=self.enabled,action_count=self.action_count,
                    early_steps=self.early_steps,probability=self.probability,
                    overrides=self.overrides,rng=copy.deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self,state):
        expected=(self.enabled,self.action_count,self.early_steps,self.probability)
        actual=tuple(state[k] for k in ("enabled","action_count","early_steps","probability"))
        if state["version"]!=1 or expected!=actual:
            raise ValueError("Exploration checkpoint configuration differs")
        self.rng.bit_generator.state=copy.deepcopy(state["rng"])
        self.overrides=int(state["overrides"])
