import random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

import walking_residual_env as w
import walking_residual_pilot as p


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


if __name__ == '__main__':
    unittest.main()
