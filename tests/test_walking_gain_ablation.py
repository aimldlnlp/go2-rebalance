import unittest
from unittest.mock import patch

import numpy as np

import walking_gain_ablation as a


class GainTests(unittest.TestCase):
    def test_gain_clipping_and_no_mutation(self):
        action = np.linspace(-.8,.8,12,dtype=np.float32)
        original = action.copy()
        for gain in a.GAINS:
            np.testing.assert_array_equal(a.choose_action(action,gain,None),action)
            expected = np.clip(action.astype(np.float64)*gain,-1,1)
            np.testing.assert_array_equal(a.choose_action(action,gain,0),expected)
            self.assertLessEqual(np.max(np.abs(.1*expected)),.1)
        np.testing.assert_array_equal(action,original)
        for value,gain in [(np.zeros(11),1),(np.full(12,np.nan),2),(np.zeros(12),3),
                           (np.full(12,np.inf),4)]:
            with self.assertRaises(ValueError):
                a.choose_action(value,gain,None)

    def test_intervention_waits_for_latch_and_previous_is_applied(self):
        import mujoco
        import walking_residual_env as original
        env = a.w.WalkingResidualEnv('FR_calf',record_trace=True)
        self.addCleanup(env.close)
        env.reset(seed=23000)
        proposal = np.full(12,.4,dtype=np.float32)
        selected = a.choose_action(proposal,4,env.onset)
        np.testing.assert_array_equal(selected,proposal)
        real = mujoco.mj_step
        calls = 0
        def cross(model,data):
            nonlocal calls
            real(model,data)
            calls += 1
            if calls == 1:
                data.qpos[:2] = env.origin+.51*env.forward
        with patch.object(original.mujoco,'mj_step',cross):
            obs,*_ = env.step(selected)
        self.assertEqual(env.onset,1)
        self.assertEqual(env.tick,10)
        np.testing.assert_array_equal(obs[48:60],selected.astype(np.float32))
        applied = a.choose_action(proposal,4,env.onset)
        np.testing.assert_array_equal(applied,np.ones(12))
        obs,*_ = env.step(applied)
        np.testing.assert_array_equal(env.previous,applied)
        np.testing.assert_array_equal(obs[48:60],applied.astype(np.float32))
        np.testing.assert_array_equal(env.trace()['control_residual_actions'],[selected,applied])

    def test_primary_decision_keeps_tradeoffs_inconclusive(self):
        for success,fall,result in [(1,0,'better_on_primary'),(0,-1,'better_on_primary'),
                (-1,0,'worse_on_primary'),(0,1,'worse_on_primary'),
                (0,0,'unchanged_inconclusive'),(1,1,'tradeoff_inconclusive'),(-1,-1,'tradeoff_inconclusive')]:
            self.assertEqual(a.primary_direction(success,fall),result)

    def test_prefix_detects_corruption_but_allows_postintervention_change(self):
        state = ['qpos','qvel']
        physics = ['requested_torque','limited_torque','applied_torque','actuator_force',
                   'command','target','kp','kd','contacts','strength']
        control = ['control_observations','control_raw_actions','control_actions',
                   'residual_observations','residual_policy_actions',
                   'control_residual_actions','control_rewards','reward_components']
        reference = {k:np.zeros((31,2)) for k in state+physics}
        reference.update({k:np.zeros((4,2)) for k in control})
        actual = {k:v.copy() for k,v in reference.items()}
        actual['qpos'][21] = 1
        actual['control_residual_actions'][2] = 1
        a.verify_prefix(actual,reference,20)
        for key,index in [('qpos',20),('applied_torque',19),('residual_policy_actions',2),('control_rewards',1)]:
            with self.subTest(key=key):
                bad = {k:v.copy() for k,v in actual.items()}
                bad[key][index] = 1
                with self.assertRaises(AssertionError):
                    a.verify_prefix(bad,reference,20)

    def test_report_pairs_by_pose_and_keeps_all_groups(self):
        import report_walking_gain_ablation as report
        rows = [dict(gain=gain,condition=condition,seed=seed,success=False,fallen=False,
            stall_window_start_s=None,fixed_horizon_progress_m={'0.5':None,'2.0':None},
            **{key:float(gain) for key in report.METRICS})
            for gain in a.GAINS for condition in a.CONDITIONS for seed in a.SEEDS]
        result = report.summarize(rows[::-1])
        self.assertEqual(len(result['groups']),16)
        self.assertEqual(len(result['paired']),80)
        for group in result['groups']:
            self.assertEqual(group['median_paired_progress_delta_m'],group['gain']-1)
            self.assertEqual(group['primary_direction'],'unchanged_inconclusive')
            self.assertIsNone(group['fixed_horizon']['2.0']['median_progress_m'])
        with self.assertRaises(AssertionError):
            report.summarize(rows[:-1]+[rows[0]])


if __name__ == '__main__':
    unittest.main()
