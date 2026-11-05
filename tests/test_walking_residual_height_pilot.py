import random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

import walking_residual_height_env as w
import walking_residual_height_pilot as p


class PilotTests(unittest.TestCase):
    def test_pose_pool_is_independent_of_sb3_training_seed(self):
        folder = self.enterContext(TemporaryDirectory())
        env = p.TrainingPool(w.WalkingResidualEnv('mixed'), [24000,24001], Path(folder)/'episodes.jsonl')
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=0)
        self.assertEqual(env.episode['pose_seed'], 24000)
        direct = w.WalkingResidualEnv('mixed')
        self.addCleanup(direct.close)
        expected, _ = direct.reset(seed=24000)
        np.testing.assert_array_equal(obs, expected)
        env.reset(seed=0)
        self.assertEqual(env.episode['pose_seed'], 24001)
        self.assertNotIn(env.episode['pose_seed'], range(23000,23005))

    def test_validation_preserves_global_training_rng(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        with p.preserved_rng():
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            random.random(), np.random.random(), torch.rand(3)
        got = random.random(), np.random.random(), torch.rand(3)
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        want = random.random(), np.random.random(), torch.rand(3)
        self.assertEqual(got[0], want[0])
        self.assertEqual(got[1], want[1])
        torch.testing.assert_close(got[2], want[2], rtol=0, atol=0)

    def test_wall_budget_and_nonfinite_parameters_abort(self):
        with patch.object(p.time, 'perf_counter', return_value=10.):
            with self.assertRaises(p.BudgetExpired):
                p.budget_check(10.)
            p.budget_check(11.)
        fake = type('Fake', (), {'policy':torch.nn.Linear(2,2)})()
        p.finite_policy(fake)
        with torch.no_grad():
            fake.policy.weight[0,0] = float('nan')
        with self.assertRaises(FloatingPointError):
            p.finite_policy(fake)

    def test_promotion_requires_every_condition_and_no_falls(self):
        rows = [{'condition':c,'seed':seed,'fallen':False,'interrupted':False,
            'success':c=='healthy' or seed<23003,'residual_effort':.1}
            for c in w.CONDITIONS for seed in range(23000,23005)]
        self.assertTrue(p.promotion(rows))
        self.assertFalse(p.promotion(rows[:-1]))
        rows[0]['fallen'] = True
        self.assertFalse(p.promotion(rows))
        rows[0]['fallen'] = False
        rows[0]['success'] = False
        self.assertFalse(p.promotion(rows))
        rows[0]['success'] = True
        rows[-5]['success'] = False
        self.assertFalse(p.promotion(rows))

    def test_selection_tie_prefers_less_effort_then_earlier_checkpoint(self):
        rows = [{'condition':c,'seed':seed,'fallen':False,'success':True,'residual_effort':.1}
            for c in w.CONDITIONS for seed in range(23000,23005)]
        self.assertGreater(p.selection_key(32768, rows), p.selection_key(65536, rows))
        worse = [{**r,'residual_effort':.2} for r in rows]
        self.assertGreater(p.selection_key(65536, rows), p.selection_key(32768, worse))


class HeightVariantTests(unittest.TestCase):
    def test_evaluation_trace_and_independent_reward_nominal_audit(self):
        from review_walking_residual_height_pilot import audit_reward_and_nominal
        from review_walking import verify_episode
        import sudden as s
        folder = Path(self.enterContext(TemporaryDirectory()))
        (folder/'checkpoint_32768.zip').write_bytes(b'test checkpoint hash only')
        fake = type('ZeroPolicy', (), {'predict':lambda self, obs, deterministic: (np.zeros(12,dtype=np.float32),None)})()
        with patch.object(p, 'OUT', folder):
            rows = p.evaluate(fake,'confirmation',{'validation_seeds':[20500],'checkpoint_steps':32768},float('inf'))
        self.assertEqual(len(rows),5)
        for row in rows:
            path = folder/'confirmation'/f"{row['condition']}_20500.npz"
            verify_episode(path,row,{})
            with np.load(path,allow_pickle=False) as arrays:
                audit_reward_and_nominal(arrays,row)
                broken = dict(arrays)
                broken['control_rewards'] = arrays['control_rewards']+.01
                with self.assertRaises(AssertionError):
                    audit_reward_and_nominal(broken,row)

    def test_only_height_reward_changes_under_nonzero_actions(self):
        import walking_residual_env as original
        for condition in w.CONDITIONS:
            old = original.WalkingResidualEnv(condition,record_trace=True)
            new = w.WalkingResidualEnv(condition,record_trace=True)
            self.addCleanup(old.close)
            self.addCleanup(new.close)
            obs, _ = old.reset(seed=20500)
            other, _ = new.reset(seed=20500)
            np.testing.assert_array_equal(obs,other)
            for i in range(300):
                action = .2*np.sin(np.arange(12)+i*.1)
                a, b = old.step(action), new.step(action)
                np.testing.assert_array_equal(a[0],b[0])
                self.assertEqual(a[2:4],b[2:4])
                for key in original.REWARD_WEIGHTS:
                    if key!='height_error':
                        self.assertEqual(a[4]['reward_terms'][key],b[4]['reward_terms'][key])
                self.assertEqual(b[4]['reward_terms']['height_error'],-10*(new.data.qpos[2]-.338)**2)
                self.assertEqual(b[1],sum(b[4]['reward_terms'].values()))
                if a[2] or a[3]:break
            for key, array in old.trace().items():
                np.testing.assert_array_equal(array,new.trace()[key])
            self.assertEqual(old.mission_result(),new.mission_result())

    def test_nonfinite_path_keeps_original_fall_reward(self):
        env = w.WalkingResidualEnv()
        self.addCleanup(env.close)
        env.reset(seed=20500)
        env.data.qpos[7] = np.nan
        _, reward, terminated, _, info = env.step(np.zeros(12))
        self.assertEqual(reward,-10.)
        self.assertTrue(terminated)
        self.assertEqual(info['reward_terms'],{'fall':-10.})

    def test_confirmation_gate_uses_frozen_new_cohort(self):
        from review_walking_residual_height_pilot import actual_gate
        rows = [{'condition':c,'seed':seed,'fallen':False,'interrupted':False,
            'success':c=='healthy' or seed<23303,'residual_effort':.1}
            for c in w.CONDITIONS for seed in p.CONFIRMATION_SEEDS]
        self.assertTrue(actual_gate(rows,p.CONFIRMATION_SEEDS))
        self.assertFalse(actual_gate(rows,list(range(23000,23005))))
        rows[0]['interrupted'] = True
        self.assertFalse(actual_gate(rows,p.CONFIRMATION_SEEDS))


if __name__ == '__main__':
    unittest.main()
