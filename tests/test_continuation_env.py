import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from continuation_env import ContinuationEnv

def make(**kwargs):
    config=dict(seed=23,target=20,weights={"stonepick":1.},namespace=39803,
                block_actions=8,auxiliary_horizon=4)
    config.update(kwargs)
    return ContinuationEnv(**config)

class ContinuationTest(unittest.TestCase):
    def test_resume_same_observations_rewards_and_counters(self):
        a=make();b=make()
        try:
            a.reset();a.step(0);b.load(a.save())
            for _ in range(19):
                if a.done:
                    self.assertTrue(b.done)
                    np.testing.assert_array_equal(a.reset(),b.reset())
                xa,ra,da,_=a.step(0);xb,rb,db,_=b.step(0)
                np.testing.assert_array_equal(xa,xb)
                self.assertEqual((ra,da,a.actions,a.counts),(rb,db,b.actions,b.counts))
            self.assertEqual(a.actions,20)
            with self.assertRaises(RuntimeError):a.reset()
        finally:a.close();b.close()

    def test_protocol_changes_rejected(self):
        a=make();b=make(target=21)
        try:
            with self.assertRaises(ValueError):b.load(a.save())
        finally:a.close();b.close()

    def test_cutoff_bootstraps_and_requires_reset(self):
        env=make(auxiliary_horizon=1)
        try:
            env.reset()
            _,_,done,info=env.step(0)
            self.assertTrue(done)
            self.assertEqual(info["discount"],1.)
            with self.assertRaises(RuntimeError):env.step(0)
        finally:env.close()

    def test_invalid_weights(self):
        with self.assertRaises(ValueError):make(weights={"natural":.5})
        with self.assertRaises(ValueError):make(weights={"natural":float("nan")})

if __name__=="__main__":unittest.main()
