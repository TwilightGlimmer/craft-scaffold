import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from bounded_exploration import BoundedExploration

class BoundedExplorationTest(unittest.TestCase):
    def test_matches_frozen_trial_random_schedule(self):
        for enabled in [False,True]:
            module=BoundedExploration(39711,enabled)
            reference=np.random.default_rng(39711)
            for t in range(400):
                age=t%128;last=t%47==0
                coin=float(reference.random());action=int(reference.integers(17))
                override=enabled and not last and age<32 and coin<0.5
                self.assertEqual(module.choose(4,age,last),(action if override else 4,override))

    def test_resume_matches_continuous(self):
        original=BoundedExploration(39711,True)
        for t in range(97): original.choose(2,t%128)
        state=original.state_dict()
        expected=[original.choose(2,t%128) for t in range(97,400)]
        restored=BoundedExploration(0,True)
        restored.load_state_dict(state)
        actual=[restored.choose(2,t%128) for t in range(97,400)]
        self.assertEqual(actual,expected)
        self.assertEqual(restored.overrides,original.overrides)
        self.assertEqual(restored.state_dict(),original.state_dict())

    def test_terminal_and_late_actions_not_overridden(self):
        module=BoundedExploration(1,True,probability=1)
        self.assertEqual(module.choose(5,0,True),(5,False))
        self.assertEqual(module.choose(5,32),(5,False))

    def test_reject_changed_protocol_on_resume(self):
        module=BoundedExploration(1,True)
        state=module.state_dict()
        with self.assertRaises(ValueError):
            BoundedExploration(1,False).load_state_dict(state)

if __name__=="__main__": unittest.main()
