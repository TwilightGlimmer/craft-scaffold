"""Pixel-only Dreamer adapter for explicit continuation protocols."""
import pickle
from embodied.envs.crafter import Crafter
from continuation_env import ContinuationEnv
from legacy_native_state import load_legacy_native_state
from bounded_exploration import BoundedExploration

class ContinuationCrafter(Crafter):
    def __init__(self,protocol,size=(64,64)):
        self.protocol=protocol
        super().__init__("reward",size=size,logs=True,seed=protocol["seed"])
        self._env.close()
        self._env=ContinuationEnv(seed=protocol["seed"],target=protocol["target"],
            weights=protocol["weights"],namespace=protocol["namespace"],
            block_actions=protocol["block_actions"],
            auxiliary_horizon=protocol["auxiliary_horizon"],size=size,reward=True)
        self._env.settings["exploration"]=dict(protocol["exploration"])
        self._env.exploration=BoundedExploration(**protocol["exploration"])

    def save(self):
        return pickle.dumps(dict(environment=self._env.save(),episode=self._episode,
            length=self._length,reward=self._reward,done=self._done),pickle.HIGHEST_PROTOCOL)

    def load(self,data):
        state=pickle.loads(data)
        environment=load_legacy_native_state(state["environment"])
        if "settings" in environment:
            self._env.load(state["environment"])
        else:
            self._env.migrate_legacy(state["environment"],self.protocol["source_step"])
        self._episode=state["episode"];self._length=state["length"]
        self._reward=state["reward"];self._done=state["done"]

    def close(self):self._env.close()
