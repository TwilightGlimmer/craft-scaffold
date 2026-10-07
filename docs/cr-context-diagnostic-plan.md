# CR context diagnosis: design, not launched

The next test will measure both the frozen 600k source and the CR 610k candidate on the same complete recorded histories around all 13 real target successes (11 stone pickaxes, 2 iron pickaxes). No gradients, reward changes, replay insertion, action masks or new training are involved.

Earlier baseline diagnostics are not measurements of the CR candidate. They showed a strong context effect: natural first-time wood-pickaxe rewards were predicted near 0.927 in four selected successful events, but resetting recurrent history at those same observations reduced the prediction near zero. Artificially resetting history is an out-of-distribution diagnostic manipulation, not evidence of a training reset bug. The earlier prepared stone-pickaxe cases likewise cannot establish that the CR candidate still underpredicts rewards after adaptation.

The bounded follow-up therefore preserves recorded history from the episode's real initial observation. It compares immediate prior reward (eight seeded draws), posterior reward/loss after the actual success frame, and actor target probability/rank. State critic values may be recorded but are explicitly V values, not action values or counterfactual optimal returns. At most 13 earlier feasible states without a target command provide descriptive controls.

The two checkpoints see identical histories and actual actions. Improved prediction without improved actor probability would motivate policy/value diagnostics; persistent posterior error would motivate representation/reward fitting diagnostics. Neither is sufficient causal proof. The bank is selected for success and must never be reported as a success-rate evaluation.

Budget: one GPU, two CPU cores, 20 GiB VRAM, 16 GiB process-tree RAM, 20 minutes total and 480 seconds per model, at most 26 states. Freeze source/weights/bank and verify the action-carry interface before inference. Stop on failure; no automatic retries or training promotion.

Status: protocol prepared; inference has not started.
