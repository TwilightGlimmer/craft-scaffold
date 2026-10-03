"""Training-only curriculum wrapper. Metadata in info must not enter the agent."""
import pickle
import numpy as np
import crafter

METHODS = ('natural', 'fixed_mix', 'fading_mix')
BLOCK_ACTIONS = 2500
TOTAL_ACTIONS = 300000

def assisted_block(method, block):
    if method not in METHODS:
        raise ValueError(method)
    if not 0 <= block < 120:
        raise ValueError(block)
    return (method == 'fixed_mix' and block % 4 == 0 or
            method == 'fading_mix' and block < 60 and block % 2 == 0)


class StableCrafterEnv(crafter.Env):
    # Uniform sampling is unchanged; a stable ordering removes object-address
    # dependence from the mapping between RNG draws and despawn candidates.
    def _balance_object(self, chunk, objs, *args, **kwargs):
        ordered = sorted(objs, key=lambda obj: (
            int(obj.pos[0]), int(obj.pos[1]), type(obj).__name__))
        return super()._balance_object(chunk, ordered, *args, **kwargs)

class CurriculumEnv:
    def __init__(self, method, seed, namespace=3001, **kwargs):
        if method not in METHODS:
            raise ValueError(method)
        self.method, self.seed, self.namespace = method, int(seed), int(namespace)
        self.kwargs = kwargs
        self.actions = self.assisted_actions = self.resets = 0
        self.done = True
        self.env = StableCrafterEnv(seed=seed, **kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.last_reset_audit = None

    def reset(self):
        if not self.done:
            raise RuntimeError('Reset would discard an active episode')
        if self.actions >= TOTAL_ACTIONS:
            raise RuntimeError('Training budget exhausted')
        block = self.actions // BLOCK_ACTIONS
        assisted = assisted_block(self.method, block)
        seed = int(np.random.SeedSequence(
            [self.namespace, self.seed, self.resets]).generate_state(1)[0] % (2**31-1))
        self.env = StableCrafterEnv(seed=seed, **self.kwargs)
        image = self.env.reset()
        self.resets += 1
        audit = dict(seed=seed, block=block, assisted=assisted,
                     actions=self.actions, reset_index=self.resets-1)
        if assisted:
            player, world = self.env._player, self.env._world
            target = (int(player.pos[0])+1, int(player.pos[1]))
            material, occupant = world[target]
            audit.update(replaced_material=material,
                         removed_object=type(occupant).__name__ if occupant else None,
                         wood_before=int(player.inventory['wood']))
            if occupant is not None:
                world.remove(occupant)
            world[target] = 'table'
            player.inventory['wood'] = 1
            image = self.env._obs()
        self.last_reset_audit = audit
        self.done = False
        return image

    def step(self, action):
        if self.done:
            raise RuntimeError('reset required')
        if self.actions >= TOTAL_ACTIONS:
            raise RuntimeError('Training budget exhausted')
        assisted = assisted_block(self.method, self.actions // BLOCK_ACTIONS)
        image, reward, done, info = self.env.step(action)
        self.actions += 1
        self.assisted_actions += int(assisted)
        boundary = self.actions % BLOCK_ACTIONS == 0
        if boundary and not done:
            done = True
            info = dict(info, discount=1.0)
        self.done = bool(done)
        info = dict(info, curriculum_audit=dict(
            actions=self.actions, assisted_actions=self.assisted_actions,
            assisted=assisted, block_boundary=boundary, resets=self.resets))
        return image, reward, self.done, info

    def save(self):
        state=dict(self.__dict__)
        state.pop('step',None)  # Runtime instrumentation is reattached after load.
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, state):
        state = pickle.loads(state)
        if (state['method'], state['seed'], state['namespace']) != (
                self.method, self.seed, self.namespace):
            raise ValueError('Incompatible curriculum checkpoint')
        self.__dict__.update(state)

    def close(self):
        self.env.close()
