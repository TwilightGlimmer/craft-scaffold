"""Experimental CR-inspired sequence selector; not connected to training.

Priority uses the published count-plus-model-loss expression. Sequence means,
duplicate-loss averaging and migration counts are explicit local choices.
"""
import collections
import pickle
import threading
import numpy as np
from embodied.core.selectors import Prioritized

class CuriositySelector(Prioritized):
    def __init__(self,*,c,beta,alpha,epsilon,initial_loss,seed):
        values=[c,beta,alpha,epsilon,initial_loss]
        if not all(np.isfinite(x) for x in values):
            raise ValueError("Nonfinite priority settings")
        if c<0 or not 0<beta<=1 or alpha<=0 or epsilon<=0 or initial_loss<0:
            raise ValueError("Invalid priority settings")
        self.signature=tuple(float(x) for x in values)
        self.c,self.beta,self.alpha,self.epsilon,self.initial_loss=self.signature
        initial=self.c+(self.initial_loss+self.epsilon)**self.alpha
        super().__init__(exponent=1.,initial=initial,seed=seed)
        self.lock=threading.RLock()
        self.counts={};self.model_losses={}
        self.sample_count=0;self.last_sample=None

    @staticmethod
    def _ids(stepids):
        return [x if isinstance(x,bytes) else x.tobytes() for x in stepids]

    def _score(self,stepid):
        return self.c*self.beta**self.counts[stepid]+(self.model_losses[stepid]+self.epsilon)**self.alpha

    def __setitem__(self,key,stepids):
        ids=self._ids(stepids)
        if not ids or len(set(ids))!=len(ids):
            raise ValueError("A sequence requires unique step IDs")
        with self.lock:
            if key in self.items:raise ValueError("Duplicate sequence key")
            for sid in ids:
                self.counts.setdefault(sid,0)
                self.model_losses.setdefault(sid,self.initial_loss)
                self.prios[sid]=self._score(sid)
            super().__setitem__(key,ids)

    def __call__(self):
        with self.lock:
            key=self.tree.sample()
            probability=self.tree.entries[key].uprob/self.tree.root.uprob
            ids=self.items[key]
            for sid in ids:self.counts[sid]+=1
            super().prioritize(ids,[self._score(sid) for sid in ids])
            self.sample_count+=1
            self.last_sample=dict(key=key,probability=float(probability),sample_count=self.sample_count)
            return key

    def prioritize(self,stepids,losses):
        ids=self._ids(stepids)
        losses=np.asarray(losses,dtype=np.float64)
        if losses.shape!=(len(ids),) or not np.isfinite(losses).all() or np.any(losses<0):
            raise ValueError("Expected one finite nonnegative model loss per step ID")
        with self.lock:
            grouped=collections.defaultdict(list)
            for sid,loss in zip(ids,losses):
                if sid in self.stepitems:grouped[sid].append(float(loss))
            # Repeated IDs in overlapping sampled sequences have one mean loss.
            for sid,values in grouped.items():self.model_losses[sid]=sum(values)/len(values)
            if grouped:
                keys=list(grouped)
                super().prioritize(keys,[self._score(sid) for sid in keys])

    def __delitem__(self,key):
        with self.lock:
            ids=list(self.items[key])
            super().__delitem__(key)
            for sid in ids:
                if sid not in self.stepitems:
                    self.counts.pop(sid,None);self.model_losses.pop(sid,None)

    def save(self):
        with self.lock:
            state=dict(version=1,signature=self.signature,tree=self.tree,
                       prios=dict(self.prios),stepitems=dict(self.stepitems),items=self.items,
                       counts=self.counts,model_losses=self.model_losses,
                       sample_count=self.sample_count,last_sample=self.last_sample)
            return pickle.dumps(state,pickle.HIGHEST_PROTOCOL)

    def load(self,data):
        state=pickle.loads(data)  # Trusted local checkpoint only.
        if state["version"]!=1 or state["signature"]!=self.signature:
            raise ValueError("Curiosity selector protocol changed")
        if set(state["tree"].entries)!=set(state["items"]):
            raise ValueError("Sampler tree and items disagree")
        with self.lock:
            for field in ("tree","items","counts","model_losses","sample_count","last_sample"):
                setattr(self,field,state[field])
            self.prios=collections.defaultdict(lambda:self.initial,state["prios"])
            self.stepitems=collections.defaultdict(list,state["stepitems"])
