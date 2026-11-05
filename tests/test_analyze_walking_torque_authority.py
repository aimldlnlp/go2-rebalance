import unittest
import numpy as np
import analyze_walking_torque_authority as a


class Tests(unittest.TestCase):
    def test_box_bounds_random_actions_and_asymmetric_speed_limit(self):
        rng=np.random.default_rng(7)
        n=200
        nominal=rng.uniform(a.RANGES[:,0],a.RANGES[:,1],size=(n,12))
        q=nominal+rng.normal(0,.4,(n,12))
        v=rng.uniform(-35,35,(n,12))
        kp,kd=np.full((n,12),25.),np.full((n,12),.5)
        strength=np.ones((n,12));strength[:,11]=.1
        low,high=a.envelope(nominal,q,v,kp,kd,strength)
        for _ in range(30):
            target=np.clip(nominal+.1*rng.uniform(-1,1,(n,12)),a.RANGES[:,0],a.RANGES[:,1])
            actual=a.torque_limit(kp*(target-q)-kd*v,v,'go2_rl_lab',a.LIMITS)*strength
            self.assertTrue(np.all(actual>=low-a.EPS) and np.all(actual<=high+a.EPS))
        low_full,high_full=a.envelope(nominal,q,v,kp,kd,np.ones((n,12)))
        np.testing.assert_array_equal(low[:,11],.1*low_full[:,11])
        np.testing.assert_array_equal(high[:,11],.1*high_full[:,11])

    def test_saturated_and_free_joint(self):
        nominal=np.zeros((1,12));q=np.zeros((1,12));v=np.zeros((1,12))
        nominal[:]=np.mean(a.RANGES,axis=1)
        q[:]=nominal
        kp,kd=np.full((1,12),25.),np.full((1,12),.5)
        lo,hi=a.envelope(nominal,q,v,kp,kd,np.ones((1,12)))
        np.testing.assert_allclose(hi-lo,5.,atol=1e-12)
        q-=10
        lo,hi=a.envelope(nominal,q,v,kp,kd,np.ones((1,12)))
        np.testing.assert_array_equal(lo,hi)
        self.assertTrue(np.all(lo>0))


if __name__=='__main__':
    unittest.main()
