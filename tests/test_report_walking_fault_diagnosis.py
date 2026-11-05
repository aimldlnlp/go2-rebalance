import unittest
import numpy as np
from report_walking_fault_diagnosis import joint_metrics


class JointTests(unittest.TestCase):
    def test_signed_bounds_and_onset_exclusion(self):
        actions=np.zeros((3,12));actions[1,0]=1.;actions[2,0]=-1.
        requested=np.zeros((30,12));requested[20:,2]=1.
        a={'qpos':np.zeros((31,19)),'control_residual_actions':actions,
            'requested_torque':requested,'limited_torque':np.zeros_like(requested),
            'contacts':np.ones((30,4),dtype=bool)}
        r=joint_metrics(a,10)
        self.assertEqual(r['residual_bound_fraction'][0],1.)
        self.assertEqual(r['positive_bound_fraction'][0],.5)
        self.assertEqual(r['negative_bound_fraction'][0],.5)
        self.assertAlmostEqual(r['residual_rms_rad'][0],.1)
        self.assertEqual(r['torque_limit_fraction'][2],.5)
        self.assertEqual(r['all_four_contacts_fraction'],1.)


if __name__=='__main__':unittest.main()
