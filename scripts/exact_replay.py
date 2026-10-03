"""Exact in-memory uniform replay snapshot; call only with sampling stopped."""
import pickle
import embodied
import elements
import copyreg
# UUID caches Python's process-randomized bytes hash. Never persist that cache.
def _uuid_setstate(self,state):
    slots=state[1] if isinstance(state,tuple) else state
    self.value=slots['value']
    self._hash=hash(self.value)
elements.UUID.__setstate__=_uuid_setstate
copyreg.pickle(elements.UUID,lambda obj:(elements.UUID,(obj.value,)))


class ExactReplay(embodied.replay.Replay):
    FIELDS=('chunks','refs','items','fifo','itemid','current','streams','metrics')
    def __init__(self,*args,**kwargs):
        if kwargs.get('directory') or kwargs.get('online') or kwargs.get('selector'):
            raise ValueError('ExactReplay requires in-memory uniform offline replay')
        super().__init__(*args,**kwargs)
    def save(self):
        with self.rwlock.writing:
            with self.sampler.lock:
                state={k:getattr(self,k) for k in self.FIELDS}
                state['sampler']={k:getattr(self.sampler,k) for k in ('indices','keys','rng')}
                state['signature']=(self.length,self.capacity,self.chunksize)
                # Materialize a consistent copy before checkpoint I/O begins.
                return pickle.dumps(state,protocol=pickle.HIGHEST_PROTOCOL)
    def load(self,data):
        state=pickle.loads(data)
        assert state.pop('signature')==(self.length,self.capacity,self.chunksize)
        sampler=state.pop('sampler')
        with self.rwlock.writing:
            for key,value in state.items():setattr(self,key,value)
            for key,value in sampler.items():setattr(self.sampler,key,value)
