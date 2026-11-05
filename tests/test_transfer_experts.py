from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO

import transfer_experts as t


class TransferTests(unittest.TestCase):
    def test_gate_uses_only_ordinary_gravity_and_seed_splits(self):
        obs = np.zeros((4, 60), dtype=np.float32)
        obs[:, 2] = -np.cos(np.deg2rad([0., 2., 2.5, 3.5]))
        np.testing.assert_allclose(t.gates(obs), [0, 0, .5, 1], atol=.0001)
        starts = [s for seed in range(3) for r in range(t.ROUNDS) for s in t.rollout_seeds(seed, r)]
        self.assertEqual(len(starts), len(set(starts)))
        self.assertFalse(set(starts) & set(range(700, 805)))

    def test_actor_fit_reduces_loss_without_changing_critic_or_std(self):
        torch.set_num_threads(1)
        env = t.m.MultiMotorBalance()
        agent = PPO('MlpPolicy', env, seed=987654, device='cpu',
                    policy_kwargs={'net_arch': [64, 64], 'log_std_init': -2.})
        obs, _ = env.reset(seed=987654)
        data = {'observations': np.repeat(obs[None], 64, axis=0),
                'targets': np.full((64, 12), .2, dtype=np.float32), 'gates': np.ones(64, dtype=np.float32)}
        unchanged = {n: p.detach().clone() for n, p in agent.policy.named_parameters()
                     if n.startswith(('mlp_extractor.value_net', 'value_net', 'log_std'))}
        losses = t.fit(agent, data, 10, 32, np.random.default_rng(1))
        self.assertLess(losses[-1], losses[0])
        for name, before in unchanged.items():
            self.assertTrue(torch.equal(before, dict(agent.policy.named_parameters())[name]))
        self.assertEqual(agent.observation_space.shape, (60,))
        self.assertEqual(agent.num_timesteps, 0)
        with tempfile.TemporaryDirectory() as directory:
            agent.save(Path(directory) / 'student.zip')
            restored = PPO.load(Path(directory) / 'student.zip', device='cpu')
            np.testing.assert_array_equal(agent.predict(obs, deterministic=True)[0], restored.predict(obs, deterministic=True)[0])
        env.close()

    def test_no_final_evaluation_if_any_student_fails(self):
        results = {str(s): {'passed': True} for s in range(3)}
        results['1']['passed'] = False
        with patch.object(t.m, 'evaluate') as evaluate:
            with self.assertRaisesRegex(AssertionError, 'all three'):
                t.final_test(results, {})
            evaluate.assert_not_called()

    def test_prepare_and_run_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(t, 'OUT', Path(directory)):
            with self.assertRaisesRegex(AssertionError, 'overwrite'):
                t.prepare()
            (Path(directory) / 'execution.json').write_text('{}')
            with self.assertRaisesRegex(AssertionError, 'rerun'):
                t.run()


if __name__ == '__main__':
    unittest.main()
