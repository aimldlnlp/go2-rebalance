import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import expert_selector as e


class SelectorTests(unittest.TestCase):
    def test_routes_fixed_expert_and_healthy_zero_for_entire_episode(self):
        agents = {seed: Mock() for seed in range(3)}
        for seed, agent in agents.items():
            agent.predict.return_value = (np.full(12, seed + .1), None)
        policies = {c: None if seed is None else agents[seed] for c, seed in e.MAPPING.items()}
        actions = {c: [] for c in e.MAPPING}

        class Env:
            def __init__(self, motor, **kwargs):
                self.motor = motor
                self.horizon = 2
                self.data = SimpleNamespace(qpos=np.zeros(19), qvel=np.zeros(18), ctrl=np.zeros(12))

            def reset(self, seed):
                self.samples = [(0, 0., 1., .27, False, 1., 1.), (2000, 4., 1., .27, False, 1., 1.)]
                return np.zeros(60, dtype=np.float32), {}

            def step(self, action):
                actions[self.motor].append(action.copy())
                tick = 2501 + len(actions[self.motor])
                self.samples.append((tick, tick * .002, 1., .27, False, .1, .1))
                return np.zeros(60, dtype=np.float32), 0., False, False, {}

            def close(self):
                pass

        p = {"conditions": list(e.MAPPING), "diagnostic_seeds": [42], "environment": {}}
        with tempfile.TemporaryDirectory() as directory, patch.object(e.m, "MultiMotorBalance", Env):
            rows = e.evaluate(policies, Path(directory) / "selector", p)
            self.assertEqual(len(rows), 5)
            with self.assertRaises(FileExistsError):
                e.evaluate(policies, Path(directory) / "selector", p)
        for c, seed in e.MAPPING.items():
            self.assertEqual(len(actions[c]), 2)
            for action in actions[c]:
                np.testing.assert_array_equal(action, np.zeros(12) if seed is None else np.full(12, seed + .1))
        self.assertEqual([agents[s].predict.call_count for s in range(3)], [2, 2, 4])
        for agent in agents.values():
            self.assertTrue(all(call.kwargs == {"deterministic": True} for call in agent.predict.call_args_list))

    def test_prepare_and_run_reject_overwrite_before_any_simulation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(e, "OUT", Path(directory)):
            with self.assertRaisesRegex(AssertionError, "overwrite"):
                e.prepare()
            (Path(directory) / "execution.json").write_text("{}")
            with self.assertRaisesRegex(AssertionError, "rerun"):
                e.run()


if __name__ == "__main__":
    unittest.main()
