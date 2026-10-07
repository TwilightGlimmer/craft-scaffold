"""Experimental exact replay with CR-inspired sampling; trusted checkpoints only.

Stop insertion/sampling before snapshots or migration. All sampler draws,
including report/eval draws, count as replay uses. No trainer integration yet.
"""
import pickle
from exact_replay import ExactReplay
from curiosity_selector import CuriositySelector

class CuriosityReplay(ExactReplay):
    def __init__(self, *args, curiosity, **kwargs):
        super().__init__(*args, **kwargs)
        self.curiosity = dict(curiosity)
        # Replay uses selector-or-Uniform: an empty selector is falsey.
        self.sampler = CuriositySelector(**self.curiosity)

    def save(self):
        with self.rwlock.writing:
            state = {k: getattr(self, k) for k in self.FIELDS}
            state.update(signature=(self.length,self.capacity,self.chunksize),
                         sampler_kind="curiosity-v1",sampler=self.sampler.save())
            return pickle.dumps(state,pickle.HIGHEST_PROTOCOL)

    def load(self, data):
        state = pickle.loads(data)
        if state.get("sampler_kind") != "curiosity-v1":
            raise ValueError("Uniform migration must be explicit")
        if state["signature"] != (self.length,self.capacity,self.chunksize):
            raise ValueError("Replay shape changed")
        sampler = CuriositySelector(**self.curiosity)
        sampler.load(state["sampler"])
        if set(sampler.items) != set(state["items"]):
            raise ValueError("Replay and sampler items disagree")
        with self.rwlock.writing:
            for key in self.FIELDS: setattr(self,key,state[key])
            self.sampler = sampler

    def migrate_uniform(self, data):
        """Preserve experience; initialize new CR counts/losses/RNG explicitly."""
        state = pickle.loads(data)
        if "sampler_kind" in state:
            raise ValueError("Expected uniform ExactReplay snapshot")
        if state["signature"] != (self.length,self.capacity,self.chunksize):
            raise ValueError("Replay shape changed")
        order = state["sampler"]["keys"]
        if len(order) != len(set(order)) or set(order) != set(state["items"]):
            raise ValueError("Uniform sampler and replay disagree")
        # Validate/build in a temporary replay so migration errors are atomic.
        staging = ExactReplay(length=self.length,capacity=self.capacity,chunksize=self.chunksize)
        staging.load(data)
        sampler = CuriositySelector(**self.curiosity)
        for key in order:
            chunk,index = staging.items[key]
            sampler[key] = staging._getseq(chunk,index,["stepid"])["stepid"]
        with self.rwlock.writing:
            for key in self.FIELDS: setattr(self,key,getattr(staging,key))
            self.sampler = sampler
        return dict(sequences=len(self),resident_steps=len(sampler.counts),
                    previous_replay_preserved=True,counts_initialized_to_zero=True,
                    model_losses_initialized_to=self.curiosity["initial_loss"],
                    sampler_rng_reinitialized=True)
