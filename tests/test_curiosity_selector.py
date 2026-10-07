import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"dreamerv3")]
from curiosity_selector import CuriositySelector

def make(seed=7):
    return CuriositySelector(c=2,beta=.5,alpha=1,epsilon=.01,initial_loss=1,seed=seed)

class CuriositySelectorTest(unittest.TestCase):
    def test_count_decay_and_loss_priority(self):
        s=make();s[0]=[b"a"]
        self.assertAlmostEqual(s.prios[b"a"],3.01)
        s();self.assertAlmostEqual(s.prios[b"a"],2.01)
        s.prioritize([b"a"],[4]);self.assertAlmostEqual(s.prios[b"a"],5.01)
        self.assertEqual(s.last_sample["probability"],1.)

    def test_overlap_counts_are_per_sample(self):
        s=make();s[0]=[b"a",b"b"];s[1]=[b"b",b"c"]
        key=s()
        self.assertEqual(s.counts[b"b"],1)
        for sid in [b"a",b"c"]:
            self.assertEqual(s.counts[sid],int(sid in s.items[key]))

    def test_duplicate_loss_ids_average(self):
        s=make();s[0]=[b"a",b"b"]
        s.prioritize([b"a",b"a"],[2,4])
        self.assertEqual(s.model_losses[b"a"],3)

    def test_removed_ids_do_not_return(self):
        s=make();s[0]=[b"a",b"b"];s[1]=[b"b",b"c"]
        del s[0]
        s.prioritize([b"a"],[10])
        self.assertNotIn(b"a",s.prios)
        self.assertNotIn(b"a",s.counts)
        self.assertNotIn(b"a",s.stepitems)
        self.assertIn(b"b",s.counts)

    def test_exact_resume(self):
        s=make()
        for i in range(20):s[i]=[bytes([i]),bytes([i+1])]
        for _ in range(10):s()
        restored=make(99);restored.load(s.save())
        for _ in range(100):
            self.assertEqual(s(),restored())
            self.assertEqual(s.last_sample,restored.last_sample)
            self.assertEqual(s.counts,restored.counts)
        self.assertEqual(dict(s.prios),dict(restored.prios))

    def test_reject_invalid_loss_and_changed_protocol(self):
        s=make();s[0]=[b"a"]
        with self.assertRaises(ValueError):s.prioritize([b"a"],[-1])
        changed=CuriositySelector(c=3,beta=.5,alpha=1,epsilon=.01,initial_loss=1,seed=7)
        with self.assertRaises(ValueError):changed.load(s.save())

if __name__=="__main__":unittest.main()
