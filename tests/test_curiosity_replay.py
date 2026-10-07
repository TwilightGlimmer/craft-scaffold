import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"dreamerv3")]
from curiosity_replay import CuriosityReplay
from curiosity_selector import CuriositySelector
from exact_replay import ExactReplay

CFG=dict(c=2,beta=.5,alpha=1,epsilon=.01,initial_loss=1,seed=7)
def make(**overrides):
    return CuriosityReplay(length=4,capacity=12,chunksize=5,curiosity=dict(CFG,**overrides))
def add(r,i):
    r.add(dict(value=np.int32(i),is_first=np.bool_(i%7==0),
               is_last=np.bool_(i%7==6),is_terminal=np.bool_(False)))

class CuriosityReplayTest(unittest.TestCase):
    def test_not_silently_uniform(self):
        self.assertIsInstance(make().sampler,CuriositySelector)

    def test_restore_batches_and_feedback_exactly(self):
        a=make()
        for i in range(20):add(a,i)
        batch=a.sample(3)
        a.update(dict(stepid=batch["stepid"],priority=np.full((3,4),4.)))
        b=make(seed=99);b.load(a.save())
        for _ in range(10):
            x,y=a.sample(3),b.sample(3)
            for k in x:np.testing.assert_array_equal(x[k],y[k])
            for r,z in [(a,x),(b,y)]:
                r.update(dict(stepid=z["stepid"],priority=np.abs(z["value"]).astype(float)))
            self.assertEqual(a.sampler.counts,b.sampler.counts)
        self.assertEqual(dict(a.sampler.prios),dict(b.sampler.prios))

    def test_eviction_and_pending_stream_restore(self):
        a=make()
        for i in range(22):add(a,i)
        b=make();b.load(a.save())
        # Different fresh chunk UUIDs are valid; compare transition values.
        for i in range(22,50):add(a,i);add(b,i)
        self.assertEqual(len(a),12)
        self.assertEqual(set(a.items),set(a.sampler.items))
        self.assertEqual(set(a.sampler.counts),set(a.sampler.stepitems))
        for _ in range(5):
            np.testing.assert_array_equal(a.sample(2)["value"],b.sample(2)["value"])

    def test_explicit_uniform_migration_preserves_sequences(self):
        old=ExactReplay(length=4,capacity=12,chunksize=5,seed=3)
        for i in range(20):add(old,i)
        data=old.save();new=make()
        with self.assertRaises(ValueError):new.load(data)
        report=new.migrate_uniform(data)
        self.assertEqual(report["sequences"],12)
        self.assertTrue(all(x==0 for x in new.sampler.counts.values()))
        self.assertEqual(list(old.fifo),list(new.fifo))
        for key,(chunk,index) in old.items.items():
            x=old._getseq(chunk,index);y=new._getseq(*new.items[key])
            for k in x:np.testing.assert_array_equal(x[k],y[k])
        restored=make();restored.load(new.save())
        np.testing.assert_array_equal(new.sample(4)["stepid"],restored.sample(4)["stepid"])

    def test_reject_priority_protocol_change(self):
        a=make()
        for i in range(8):add(a,i)
        with self.assertRaises(ValueError):make(c=3).load(a.save())
if __name__=="__main__":unittest.main()
