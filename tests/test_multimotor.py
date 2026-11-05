import unittest

import numpy as np

import multimotor as m


class MultiMotorTests(unittest.TestCase):
    def test_randomization_covers_conditions_and_replays(self):
        env = m.MultiMotorBalance(randomize=True)
        cases, strengths = set(), set()
        for seed in range(1000, 1150):
            observation, info = env.reset(seed=seed)
            cases.add(info["fault_motor"])
            strengths.add(info["final_strength"])
            self.assertEqual(observation.shape, (60,))
            self.assertTrue(3 <= info["fault_onset_s"] <= 8)
        self.assertEqual(cases, set(m.CONDITIONS))
        self.assertEqual(strengths, {1, .1, .25, .5, .75})
        first, info1 = env.reset(seed=987652)
        second, info2 = env.reset(seed=987652)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(info1, info2)
        env.close()

    def test_fault_scales_only_selected_actual_torque(self):
        for motor in m.CONDITIONS:
            env = m.MultiMotorBalance(motor=motor)
            env.reset(seed=987652)
            env.frame_skip = 1
            env.data.time = 5
            expected = np.clip(50 * (env.home - env.data.qpos[env.qadr]) - 3 * env.data.qvel[env.vadr],
                               env.limits[:, 0], env.limits[:, 1])
            env.step(np.zeros(12))
            if motor != "healthy":
                expected[env.weak_motor] *= .1
            np.testing.assert_array_equal(env.data.ctrl, expected)
            env.close()

    def test_frozen_four_candidates_and_holdout(self):
        p = m.verify()
        self.assertEqual(list(p["configurations"]), list("ABCD"))
        self.assertEqual(p["ppo"]["steps"], 262144)
        self.assertEqual(p["training_seeds"], [0, 1, 2])
        self.assertEqual(p["validation_seeds"], list(range(100, 105)))
        self.assertEqual(p["final_seeds"], list(range(600, 620)))
        self.assertFalse(set(p["final_seeds"]) & set(p["prior_used_seeds"]))

    def test_acceptance_catches_motor_failure_and_pairing(self):
        pd = [{"condition": c, "seed": seed, "initial_state_sha256": str(seed),
               "fallen": False, "simulated_s": 30, "mean_tilt_deg": 1,
               "mean_post_tilt_deg": 5, "max_tilt_deg": 10} for c in m.CONDITIONS for seed in [10, 11]]
        learned = [dict(r, mean_tilt_deg=1.3, mean_post_tilt_deg=3) for r in pd]
        self.assertTrue(m.evidence(pd, learned, [10, 11])["passed"])
        learned[-1]["max_tilt_deg"] += 1e-8
        self.assertFalse(m.evidence(pd, learned, [10, 11])["checks"]["RR_calf_peak"])
        learned[-1]["max_tilt_deg"] = 10
        learned[-1]["mean_post_tilt_deg"] += 1e-8
        self.assertFalse(m.evidence(pd, learned, [10, 11])["checks"]["RR_calf_mean"])
        learned[0]["mean_tilt_deg"] += 1e-8
        self.assertFalse(m.evidence(pd, learned, [10, 11])["checks"]["healthy"])
        learned[-1]["simulated_s"] = 29
        self.assertFalse(m.evidence(pd, learned, [10, 11])["checks"]["survival"])
        learned[-1]["initial_state_sha256"] = "wrong"
        with self.assertRaises(AssertionError):
            m.evidence(pd, learned, [10, 11])

    def test_old_front_left_evaluation_is_preserved(self):
        old = m.s.SuddenBalance()
        new = m.MultiMotorBalance(motor="FL_calf", record=True)
        old.reset(seed=987652)
        new.reset(seed=987652)
        for _ in range(260):
            old.step(np.zeros(12))
            new.step(np.zeros(12))
        np.testing.assert_array_equal(old.data.qpos, new.data.qpos)
        np.testing.assert_array_equal(old.samples, new.samples)
        old.close()
        new.close()


if __name__ == "__main__":
    unittest.main()
