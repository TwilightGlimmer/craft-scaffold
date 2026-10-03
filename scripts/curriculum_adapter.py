"""Dreamer adapter. Audit metadata is never part of observation space."""
import pickle
from embodied.envs.crafter import Crafter
from curriculum_env import CurriculumEnv

class CurriculumCrafter(Crafter):
    def __init__(self, method, seed, size=(64,64)):
        super().__init__('reward', size=size, logs=True, seed=seed)
        self._env.close()
        self._env=CurriculumEnv(method, seed, size=size, reward=True)

    def save(self):
        return pickle.dumps(dict(environment=self._env.save(),
            episode=self._episode,length=self._length,reward=self._reward,
            done=self._done),protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, data):
        state=pickle.loads(data)
        self._env.load(state['environment'])
        self._episode=state['episode']
        self._length=state['length']
        self._reward=state['reward']
        self._done=state['done']

    def close(self):
        self._env.close()
