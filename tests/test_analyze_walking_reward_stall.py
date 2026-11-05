import unittest
import numpy as np
import analyze_walking_reward_stall as a


class Tests(unittest.TestCase):
    def test_quaternion_body_orientation(self):
        q = np.zeros((2,19))
        q[0,3]=1
        q[1,3]=q[1,6]=np.sqrt(.5)
        r=a.rotation(q)
        np.testing.assert_allclose(r[0],np.eye(3),atol=1e-14)
        np.testing.assert_allclose(r[1]@np.array([1,0,0]),[0,1,0],atol=1e-14)

    def test_window_complete_terminal_and_classification(self):
        z=dict(qpos=np.zeros((501,19)),qvel=np.zeros((501,18)),
            command=np.full((500,3),0.),control_rewards=np.ones(50),
            reward_components=np.ones((50,12))/12)
        z['command'][:,0]=.2
        row=dict(initial_yaw=0.)
        self.assertIsNone(a.window(z,row,1,0,1))
        value=a.window(z,row,0,0,1)
        self.assertEqual(value['control_rows'],50)
        self.assertEqual(value['commanded_low_speed_fraction'],1.)
        self.assertEqual(value['mean_reward_per_control'],1.)
        z['qvel'][:,0]=.04
        self.assertEqual(a.window(z,row,0,0,1)['commanded_low_speed_fraction'],0.)


if __name__=='__main__':
    unittest.main()
