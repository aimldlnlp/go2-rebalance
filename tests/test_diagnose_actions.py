import unittest

import numpy as np

import diagnose_actions as d


class DiagnosticTests(unittest.TestCase):
    def test_instrumentation_preserves_dynamics_and_reward(self):
        config = d.m.verify()["configurations"]["B"]
        normal = d.m.MultiMotorBalance(motor="FL_calf", record=True, **config)
        measured = d.m.MultiMotorBalance(motor="FL_calf", record=True, **config)
        normal.reset(seed=987650)
        measured.reset(seed=987650)
        recorder = d.Recorder(measured)
        for step in range(350):
            action = np.full(12, .2 * np.sin(step / 20))
            _, reward, _, _, _ = normal.step(action)
            _, actual, _, _, _ = recorder.step(action, action)
            self.assertEqual(reward, actual)
        np.testing.assert_array_equal(normal.data.qpos, measured.data.qpos)
        np.testing.assert_array_equal(normal.data.qvel, measured.data.qvel)
        np.testing.assert_array_equal(normal.samples, measured.samples)
        self.assertLessEqual(recorder.max_reward_check_error, 1e-10)
        self.assertEqual(recorder.max_torque_check_error, 0)
        normal.close()
        measured.close()

    def test_action_clipping_gate_and_invalid_input(self):
        env = d.m.MultiMotorBalance(**d.m.verify()["configurations"]["B"])
        env.reset(seed=987650)
        recorder = d.Recorder(env)
        recorder.step(np.ones(12), np.full(12, 2.0))
        row = recorder.rows[0]
        self.assertEqual(row["gate"], 0)
        np.testing.assert_array_equal(row["applied_action"], np.zeros(12))
        self.assertEqual(row["reward_terms"][4], 0)
        with self.assertRaises(ValueError):
            recorder.step(np.zeros(12), np.full(12, np.nan))
        env.close()

    def test_checkpoint_grid_and_reserved_seeds(self):
        p = d.verify()
        self.assertEqual(len(p["checkpoints"]), 10)
        self.assertEqual(p["validation_seeds"], list(range(100, 105)))
        self.assertEqual(p["episodes"], 250)
        self.assertFalse(p["training"])
        self.assertFalse(set(p["validation_seeds"]) & set(p["reserved_seeds_unused"]))
        self.assertEqual({x["budget"] for x in p["checkpoints"].values()}, {0, 262144, 524288, 1048576})

    def test_reward_terms_cover_the_actual_scalar(self):
        env = d.m.MultiMotorBalance(**d.m.verify()["configurations"]["B"])
        env.reset(seed=987650)
        recorder = d.Recorder(env)
        _, reward, _, _, _ = recorder.step(np.zeros(12), np.zeros(12))
        self.assertAlmostEqual(recorder.rows[0]["reward_terms"].sum(), reward, places=12)
        self.assertEqual(len(d.REWARD_FIELDS), 7)
        self.assertEqual(recorder.rows[0]["physics_steps"], 10)
        env.close()


if __name__ == "__main__":
    unittest.main()
