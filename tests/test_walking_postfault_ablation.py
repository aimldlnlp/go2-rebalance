import unittest
from unittest.mock import patch

import numpy as np

import walking_postfault_ablation as a


class PostfaultTests(unittest.TestCase):
    def test_gate_requires_actual_latch_and_stays_off(self):
        action=np.linspace(-1,1,12,dtype=np.float32)
        for mode,onset,off in [('always',None,False),('always',0,False),('postfault_off',None,False),('postfault_off',0,True),('postfault_off',1234,True)]:
            selected,disabled=a.choose_action(action,mode,onset)
            self.assertEqual(disabled,off)
            np.testing.assert_array_equal(selected,np.zeros(12) if off else action)
        np.testing.assert_array_equal(action,np.linspace(-1,1,12,dtype=np.float32))
        for action,mode in [(np.zeros(11),'always'),(np.full(12,np.nan),'postfault_off'),(np.zeros(12),'unknown')]:
            with self.assertRaises(ValueError):a.choose_action(action,mode,None)

    def test_primary_decision_does_not_replace_mission_with_progress(self):
        for success,falls,result in [(1,0,'off_better_primary'),(0,-1,'off_better_primary'),
            (-1,0,'off_worse_primary'),(0,1,'off_worse_primary'),
            (0,0,'primary_unchanged_inconclusive'),(1,1,'primary_tradeoff_inconclusive'),(-1,-1,'primary_tradeoff_inconclusive')]:
            self.assertEqual(a.direction(success,falls),result)

    def test_gate_preserves_physics_until_first_decision_after_fault(self):
        import walking_residual_env as original
        import mujoco
        env=a.w.WalkingResidualEnv('FR_calf',record_trace=True)
        self.addCleanup(env.close)
        env.reset(seed=20500)
        selected,off=a.choose_action(np.full(12,.2,dtype=np.float32),'postfault_off',env.onset)
        self.assertFalse(off)
        real=mujoco.mj_step
        calls=0
        def cross(model,data):
            nonlocal calls
            real(model,data);calls+=1
            if calls==1:data.qpos[0]=env.origin[0]+.51
        with patch.object(original.mujoco,'mj_step',cross):env.step(selected)
        self.assertEqual(env.onset,1)
        np.testing.assert_array_equal(env.trace()['control_residual_actions'][0],selected)
        self.assertEqual(env.tick,10)
        selected,off=a.choose_action(np.full(12,.2,dtype=np.float32),'postfault_off',env.onset)
        self.assertTrue(off)
        env.step(selected)
        np.testing.assert_array_equal(env.trace()['control_residual_actions'][1],np.zeros(12))


if __name__=='__main__':unittest.main()
