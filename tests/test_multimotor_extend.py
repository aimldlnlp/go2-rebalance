import unittest

import torch
from stable_baselines3.common.monitor import Monitor

import multimotor_extend as e


class ContinuationTests(unittest.TestCase):
    def test_budget_split_and_original_criteria(self):
        p = e.verify()
        self.assertEqual(p["configuration"], "B")
        self.assertEqual(p["total_step_checkpoints"], [524288, 1048576])
        self.assertEqual(p["initial_total_steps"], 262144)
        self.assertEqual(p["training_seeds"], [0, 1, 2])
        self.assertEqual(p["validation_seeds"], list(range(100, 105)))
        self.assertEqual(p["final_seeds"], list(range(700, 720)))
        self.assertFalse(set(p["final_seeds"]) & set(p["prior_used_seeds"]))
        original = e.m.verify()
        self.assertEqual(p["environment"], original["configurations"]["B"])
        self.assertEqual(p["criteria"], original["criteria"])
        self.assertEqual(p["ppo"], original["ppo"])

    def test_common_success_requires_all_three(self):
        results = {str(seed): {"passed": True} for seed in range(3)}
        self.assertTrue(e.all_pass(results))
        results["1"]["passed"] = False
        self.assertFalse(e.all_pass(results))
        del results["1"]
        with self.assertRaises(AssertionError):
            e.all_pass(results)

    def test_resume_retains_optimizer_and_advances_counts(self):
        p = e.verify()
        parent = e.initial_parents()[0]
        checksum = e.m.s.digest(parent / "policy.zip")
        monitor = Monitor(e.m.MultiMotorBalance(randomize=True, **p["environment"]))
        agent = e.resume(parent, 0, 262144, monitor)
        steps = e.optimizer_steps(agent)
        self.assertGreater(steps, 0)
        old = {name: parameter.detach().clone() for name, parameter in agent.policy.named_parameters()}
        agent.learn(total_timesteps=1024, reset_num_timesteps=False)
        self.assertEqual(agent.num_timesteps, 263168)
        self.assertGreater(e.optimizer_steps(agent), steps)
        self.assertTrue(any(not torch.equal(old[name], parameter) for name, parameter in agent.policy.named_parameters()))
        self.assertEqual(e.m.s.digest(parent / "policy.zip"), checksum)
        monitor.close()

    def test_resume_rejects_wrong_seed_or_budget(self):
        p = e.verify()
        monitor = Monitor(e.m.MultiMotorBalance(randomize=True, **p["environment"]))
        for seed, previous in [(1, 262144), (0, 262145)]:
            with self.assertRaises(ValueError):
                e.resume(e.initial_parents()[0], seed, previous, monitor)
        monitor.close()


if __name__ == "__main__":
    unittest.main()
