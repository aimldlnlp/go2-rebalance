import inspect
import unittest
from unittest.mock import patch

import numpy as np
import walking_navigation_hold as h
import walking_navigation as previous


class StopLatchTests(unittest.TestCase):
    def test_boundary_chatter_cannot_restart_planar_command(self):
        q = np.zeros(19)
        q[3] = 1.
        nav = h.Navigator()
        q[0] = 1.87
        self.assertGreater(nav(q, np.zeros(2), 0., 8000)[0], 0)
        q[0] = 1.881
        np.testing.assert_array_equal(nav(q, np.zeros(2), 0., 8010)[:2], np.zeros(2))
        for x in [1.8799, 1.8796, 1.88, 1.85, 1.7]:
            q[0] = x
            np.testing.assert_array_equal(nav(q, np.zeros(2), 0., 8020)[:2], np.zeros(2))
        self.assertGreater(h.Navigator()(q, np.zeros(2), 0., 8020)[0], 0)

    def test_only_episode_navigation_changed(self):
        self.assertEqual(inspect.getsource(h.episode), h.expected_episode_source())
        self.assertIs(h.summarize, previous.summarize)
        self.assertIs(h.calibrated_model, previous.calibrated_model)
        q = np.zeros(19)
        q[3] = 1.
        for tick in [0, 999, 1125, 2000]:
            np.testing.assert_array_equal(h.mission_command(q, np.zeros(2), 0., tick), previous.mission_command(q, np.zeros(2), 0., tick))
        with patch.object(h, 'TICKS', 2):
            trace, row = h.episode(h.Actor('go2_rl_lab'), 987647)
        self.assertFalse(row['success'])
        np.testing.assert_allclose(trace['actuator_force'], trace['applied_torque'], rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
