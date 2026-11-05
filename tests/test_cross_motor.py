import unittest

import numpy as np

import cross_motor as c


class CrossMotorTests(unittest.TestCase):
    def test_actual_torque_affects_only_named_motor(self):
        for motor in c.MOTORS:
            env = c.CrossMotorBalance(motor)
            env.reset(seed=987653)
            env.frame_skip = 1
            env.data.time = 5
            expected = np.clip(50 * (env.home - env.data.qpos[env.qadr]) - 3 * env.data.qvel[env.vadr],
                               env.limits[:, 0], env.limits[:, 1])
            env.step(np.zeros(12))
            mask = np.arange(12) != env.weak_motor
            np.testing.assert_array_equal(env.data.ctrl[mask], expected[mask])
            self.assertEqual(env.data.ctrl[env.weak_motor], .1 * expected[env.weak_motor])
            self.assertEqual(env.model.actuator(env.weak_motor).name, motor)
            env.close()

    def test_initial_states_match_across_motors(self):
        states = []
        for motor in c.MOTORS:
            env = c.CrossMotorBalance(motor)
            env.reset(seed=987653)
            states.append(env.data.qpos.copy())
            env.close()
        for state in states[1:]:
            np.testing.assert_array_equal(state, states[0])
        with self.assertRaises(ValueError):
            c.CrossMotorBalance("FL_calf")

    def test_fixed_protocol_and_new_seeds(self):
        p = c.verify()
        self.assertEqual(p["motors"], ["FR_calf", "RL_calf", "RR_calf"])
        self.assertEqual(p["evaluation_seeds"], list(range(500, 520)))
        self.assertFalse(set(p["evaluation_seeds"]) & set(p["previously_used_seeds"]))
        self.assertEqual(p["criteria"], c.s.read(c.s.OUT / "protocol.json")["criteria"])
        self.assertEqual(len(p["checkpoints"]), 3)
        self.assertFalse(p["execution"]["training"])


if __name__ == "__main__":
    unittest.main()
