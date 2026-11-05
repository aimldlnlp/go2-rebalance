"""Checks for abrupt onset, pairing, frozen criteria, and recovery censoring."""

import unittest

import numpy as np

import sudden as s


class SuddenTests(unittest.TestCase):
    def test_exact_switch_and_one_motor_only(self):
        env = s.SuddenBalance()
        env.reset(seed=987654)
        for tick, expected in [(2499, 1), (2500, .1), (2501, .1)]:
            env.data.time = tick * s.DT
            self.assertEqual(env.motor_strength(), expected)
        env.reset(seed=987654)
        for _ in range(250):
            env.step(np.zeros(12))
        self.assertEqual(env.samples[-1][0], 2500)
        self.assertEqual(env.samples[-1][-2:], (.1, 1.0))
        # At the switch, the unchanged PD computes identical pre-step torques.
        torque = np.clip(50 * (env.home - env.data.qpos[env.qadr]) - 3 * env.data.qvel[env.vadr],
                         env.limits[:, 0], env.limits[:, 1])
        actual = torque.copy()
        actual[env.weak_motor] *= env.motor_strength()
        np.testing.assert_array_equal(actual[np.arange(12) != env.weak_motor],
                                      torque[np.arange(12) != env.weak_motor])
        self.assertEqual(actual[env.weak_motor], .1 * torque[env.weak_motor])
        env.step(np.zeros(12))
        self.assertTrue(all(row[-1] == .1 for row in env.samples[-10:]))
        env.close()

    def test_recovery_holds_one_second_and_censors(self):
        def samples(tilts):
            return [(i, i * .002, tilt, .27, False, 1 if i < 2500 else .1,
                     1 if i <= 2500 else .1) for i, tilt in enumerate(tilts)]
        tilts = np.ones(15001)
        tilts[2501:3000] = 3
        self.assertEqual(s.summarize(samples(tilts))["recovery_s"], 1.0)
        tilts[3000:14550] = 3
        self.assertIsNone(s.summarize(samples(tilts))["recovery_s"])
        self.assertEqual(s.summarize(samples(np.ones(15001)))["recovery_s"], 0.0)

    def test_strict_paired_peak_and_mean_boundaries(self):
        pd = [{"seed": i, "initial_state_sha256": str(i), "max_tilt_deg": 10,
               "mean_post_tilt_deg": 5, "fallen": False, "simulated_s": 30,
               "recovered": False, "recovery_s": None} for i in range(20)]
        learned = [dict(r, mean_post_tilt_deg=3, max_tilt_deg=10) for r in pd]
        self.assertTrue(s.criteria(pd, learned)["passed"])
        learned[0]["max_tilt_deg"] += 1e-8
        self.assertFalse(s.criteria(pd, learned)["checks"]["every_paired_peak"])
        learned[0]["max_tilt_deg"] = 10
        learned[0]["mean_post_tilt_deg"] += 1e-8
        self.assertFalse(s.criteria(pd, learned)["checks"]["post_fault_mean"])
        learned[0]["simulated_s"] = 29
        self.assertFalse(s.criteria(pd, learned)["checks"]["survival"])
        learned[0]["initial_state_sha256"] = "different"
        with self.assertRaises(AssertionError):
            s.criteria(pd, learned)

    def test_frozen_protocol(self):
        p = s.verify_frozen()
        self.assertEqual(p["evaluation_seeds"], list(range(400, 420)))
        self.assertFalse(set(p["evaluation_seeds"]) & set(p["previously_used_seeds"]))
        self.assertEqual(p["video_seed"], 400)
        self.assertEqual(p["criteria"], {"all_episodes_30s_no_fall": True,
                         "minimum_mean_reduction": .4, "maximum_paired_peak_ratio": 1.0})
        self.assertEqual(len(p["checkpoints"]), 3)
        self.assertFalse(p["execution"]["training"])


if __name__ == "__main__":
    unittest.main()
