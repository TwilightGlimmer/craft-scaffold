"""CPU smoke checks for native-reward skill starts; no training."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from skill_starts import StableCrafterEnv,prepare_native,SKILLS,TARGETS
from crafter import constants

class SkillStartsTest(unittest.TestCase):
    def test_all_targets_are_initially_unearned_and_reachable(self):
        for skill in SKILLS:
            with self.subTest(skill=skill):
                env=StableCrafterEnv(seed=23)
                try:
                    env.reset()
                    prepare_native(env,skill)
                    target=TARGETS[skill]
                    self.assertEqual(env._player.achievements[target],0)
                    self.assertNotIn(target,env._unlocked)
                    self.assertFalse(env._auxiliary_audit['mainline_reward_filter'])
                    self.assertEqual(env._auxiliary_audit['blocked_bonus_achievements'],[])
                    action='do' if skill in ('stone','coal','iron','diamond') else target
                    env.step(constants.actions.index(action))
                    self.assertGreater(env._player.achievements[target],0)
                    self.assertIn(target,env._unlocked)
                finally:
                    env.close()

if __name__=='__main__':
    unittest.main()
