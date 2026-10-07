import os,sys,unittest
os.environ["JAX_PLATFORMS"]="cpu"
os.environ["CUDA_VISIBLE_DEVICES"]=""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"dreamerv3")]
import jax
import jax.numpy as jnp
import numpy as np
from curiosity_agent import model_priority

class PriorityTest(unittest.TestCase):
    def test_excludes_policy_value_losses(self):
        keys=("dyn","rep","rew","con","image")
        losses={k:jnp.ones((2,3)) for k in keys}
        losses.update(policy=jnp.full((2,3),999.),value=jnp.full((2,3),999.))
        scales={k:1. for k in losses}
        ids=jnp.zeros((2,3,20),dtype=jnp.uint8)
        np.testing.assert_array_equal(model_priority(losses,scales,("image",),ids),5.)

    def test_post_context_alignment_required(self):
        losses={k:jnp.ones((2,4)) for k in ("dyn","rep","rew","con","image")}
        with self.assertRaises(ValueError):
            model_priority(losses,{k:1. for k in losses},("image",),jnp.zeros((2,3,20)))

    def test_priority_does_not_change_optimization_gradient(self):
        ids=jnp.zeros((1,2,20))
        def f(x):
            losses={k:x for k in ("dyn","rep","rew","con")}
            return model_priority(losses,{k:1. for k in losses},(),ids).sum()
        np.testing.assert_array_equal(jax.grad(f)(jnp.ones((1,2))),0.)

    def test_missing_model_loss_fails_closed(self):
        with self.assertRaises(ValueError):
            model_priority({}, {}, ("image",),jnp.zeros((1,2,20)))
if __name__=="__main__":unittest.main()
