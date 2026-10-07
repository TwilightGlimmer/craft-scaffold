"""Experimental Dreamer hook returning aligned world-model replay priorities.

Not enabled by existing training entry points. Uses scaled model losses only;
actor/value objectives and imagined returns never enter replay priority.
"""
import elements
import jax
import jax.numpy as jnp
from dreamerv3.agent import Agent

def model_priority(losses, scales, reconstruction_keys, stepid):
    keys = ("dyn","rep","rew","con",*reconstruction_keys)
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate model loss key")
    shape = stepid.shape[:2]
    for key in keys:
        if key not in losses or key not in scales:
            raise ValueError("Missing model loss: " + key)
        if losses[key].shape != shape:
            raise ValueError("Model loss and post-context step IDs misaligned")
    # Clamp only roundoff/negative loss values; host replay rejects NaN/Inf.
    return jax.lax.stop_gradient(jnp.maximum(
        sum(losses[k] * scales[k] for k in keys), 0.))

class CuriousAgent(Agent):
    def train(self, carry, data):
        carry, obs, prevact, stepid = self._apply_replay_context(carry,data)
        metrics, (carry, entries, loss_outputs, mets) = self.opt(
            self.loss,carry,obs,prevact,training=True,has_aux=True)
        metrics.update(mets)
        self.slowval.update()
        keys = tuple(k for k in self.obs_space
                     if k not in ("is_first","is_last","is_terminal","reward"))
        priority = model_priority(loss_outputs["losses"],self.scales,keys,stepid)
        updates = dict(stepid=stepid,priority=priority)
        if self.config.replay_context:
            updates.update(elements.tree.flatdict(dict(
                enc=entries[0],dyn=entries[1],dec=entries[2])))
        shape = obs["is_first"].shape
        if any(x.shape[:2] != shape for x in updates.values()):
            raise ValueError("Replay update shape mismatch")
        carry = (*carry,{k:data[k][:,-1] for k in self.act_space})
        return carry,dict(replay=updates),metrics
