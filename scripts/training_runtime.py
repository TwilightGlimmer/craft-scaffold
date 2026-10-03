"""Synchronous runner state supplementary to the agent and ExactReplay."""
import pickle
import numpy as np
import jax
import elements
from embodied.jax import internal

class TrainingRuntime:
    def __init__(self,agent,driver,env,stream,ratio,carry):
        assert not driver.parallel
        self.agent,self.driver,self.env=agent,driver,env
        self.stream,self.ratio,self.carry=stream,ratio,carry
        self.episode_return=0.0
        self.episode_length=0
    def save(self):
        a=self.agent
        state=dict(
            env=self.env.save(),acts=self.driver.acts,
            policy_carry=jax.device_get(self.driver.carry),
            train_carry=jax.device_get(self.carry[0]),
            stream_index=self.stream.index,stream_current=self.stream.current,
            ratio_prev=self.ratio._prev,
            episode_return=self.episode_return,episode_length=self.episode_length,
            policy_params=jax.device_get(a.policy_params),
            pending_sync=jax.device_get(a.pending_sync) if a.pending_sync else None,
            pending_outs=jax.device_get(a.pending_outs) if a.pending_outs else None,
            pending_mets=jax.device_get(a.pending_mets) if a.pending_mets else None,
            numpy_rng=np.random.get_state())
        return pickle.dumps(state,protocol=pickle.HIGHEST_PROTOCOL)
    def load(self,data):
        s=pickle.loads(data);a=self.agent
        self.env.load(s['env']);self.driver.acts=s['acts']
        with jax.transfer_guard('allow'):
            self.driver.carry=jax.tree.map(jax.device_put,s['policy_carry'])
        self.carry[0]=internal.device_put(s['train_carry'],a.train_sharded)
        self.stream.index=s['stream_index'];self.stream.current=s['stream_current']
        self.ratio._prev=s['ratio_prev']
        self.episode_return=s['episode_return'];self.episode_length=s['episode_length']
        a.policy_params=internal.device_put(s['policy_params'],a.policy_params_sharding)
        a.pending_sync=(internal.device_put(s['pending_sync'],a.policy_params_sharding)
                        if s['pending_sync'] else None)
        # These asynchronous outputs are only consumed as host arrays next call.
        a.pending_outs=s['pending_outs'];a.pending_mets=s['pending_mets']
        np.random.set_state(s['numpy_rng'])
