import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from go2_env import Go2Balance


class BalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env = Go2Balance()

    def test_gym_contract(self):
        check_env(self.env)

    def test_seed_replays_physics(self):
        env = self.env
        states = []
        for _ in range(2):
            env.reset(seed=100)
            for _ in range(20):
                env.step(np.zeros(12))
            states.append(env.data.qpos.copy())
        np.testing.assert_array_equal(*states)

    def test_motor_fault_ramp(self):
        env = self.env
        env.reset(seed=100)
        env.strength = 0.1
        for time, expected in [(0, 1), (5, 1), (10, 0.55), (15, 0.1), (30, 0.1)]:
            env.data.time = time
            self.assertAlmostEqual(env.motor_strength(), expected)
        env.step(np.ones(12))
        self.assertLessEqual(abs(env.data.ctrl[env.weak_motor]), 45.43 * 0.1 + 1e-8)
        env.strength = 1.0

    def test_action_validation_and_clipping(self):
        env = self.env
        env.reset(seed=100)
        with self.assertRaises(ValueError):
            env.step(np.full(12, np.nan))
        with self.assertRaises(ValueError):
            env.step(np.zeros(11))
        env.step(np.full(12, 10.0))
        self.assertTrue(np.all(np.abs(env.previous) <= 1))
        self.assertTrue(np.all(env.data.ctrl <= env.limits[:, 1]))
        self.assertTrue(np.all(env.data.ctrl >= env.limits[:, 0]))

    def test_fault_changes_only_selected_motor(self):
        env = self.env
        original_step = mujoco.mj_step
        controls = []

        def capture(model, data):
            controls.append(data.ctrl.copy())
            original_step(model, data)

        first_controls = []
        with patch("go2_env.mujoco.mj_step", side_effect=capture):
            for strength in (1.0, 0.1):
                env.reset(seed=100)
                env.strength = strength
                env.data.time = 15
                controls.clear()
                env.step(np.full(12, 0.2))
                first_controls.append(controls[0])
        expected = first_controls[0].copy()
        expected[env.weak_motor] *= 0.1
        np.testing.assert_allclose(first_controls[1], expected, atol=1e-12)
        env.strength = 1.0

    def test_fall_and_timeout_are_distinct(self):
        env = self.env
        env.reset(seed=100)
        env.data.qpos[2] = 0.1
        self.assertTrue(env.metrics()["fallen"])
        env.reset(seed=100)
        env.steps = env.horizon - 1
        _, _, terminated, truncated, _ = env.step(np.zeros(12))
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_checkpoint_roundtrip(self):
        env = self.env
        agent = PPO("MlpPolicy", env, device="cpu", seed=0, n_steps=8, batch_size=8,
                    policy_kwargs={"net_arch": [16, 16]})
        observation, _ = env.reset(seed=100)
        expected = agent.predict(observation, deterministic=True)[0]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy"
            agent.save(path)
            restored = PPO.load(path, device="cpu")
            np.testing.assert_array_equal(expected, restored.predict(observation, deterministic=True)[0])


if __name__ == "__main__":
    unittest.main()
