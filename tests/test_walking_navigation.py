import unittest
from unittest.mock import patch

import numpy as np
import walking_navigation as n
import walking_calibrated as parent


class NavigationTests(unittest.TestCase):
    def test_command_caps_frames_and_stop(self):
        q = np.zeros(19)
        for yaw in [0., .7, -1.4]:
            q[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
            forward = np.array([np.cos(yaw), np.sin(yaw)])
            for distance in [2., .4, .2, .13, .119, 0.]:
                q[:2] = (2-distance)*forward
                cmd = n.mission_command(q, np.zeros(2), yaw, 2000)
                self.assertLessEqual(np.linalg.norm(cmd[:2]), .2+1e-12)
                self.assertAlmostEqual(cmd[1], 0., places=12)
                if distance <= .12:
                    np.testing.assert_array_equal(cmd[:2], np.zeros(2))
            np.testing.assert_array_equal(n.mission_command(q, np.zeros(2), yaw, 999), np.zeros(3))
        q[:2] = 0
        q[3:7] = [1, 0, 0, 0]
        self.assertAlmostEqual(n.mission_command(q, np.zeros(2), 0., 1125)[0], .1)

    def test_same_physics_and_mission_evaluator(self):
        self.assertIs(n.calibrated_model, parent.calibrated_model)
        self.assertIs(n.summarize, parent.summarize)
        self.assertIs(n.torque_limit, parent.torque_limit)
        with patch.object(n, 'TICKS', 2):
            arrays, row = n.episode(n.Actor('go2_rl_lab'), 987648)
        self.assertFalse(row['success'])
        np.testing.assert_allclose(arrays['actuator_force'], arrays['applied_torque'], rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
