import unittest
import numpy as np
import diagnose_walking_fault as d


class DiagnosisTests(unittest.TestCase):
    def test_stall_full_window_and_command_requirement(self):
        progress=np.r_[np.arange(501)*.001,np.full(1000,.5)]
        command=np.full(len(progress)-1,.2)
        start=d.stall_start(progress,command,500,500,.02)
        self.assertEqual(start,0.)
        self.assertIsNone(d.stall_start(progress,np.zeros_like(command),500,500,.02))
        self.assertIsNone(d.stall_start(progress[:700],command[:699],500,500,.02))

    def test_limiter_direction_speed_and_physical_limits(self):
        requested=np.full((4,12),100.)
        velocity=np.array([np.zeros(12),np.ones(12),np.full(12,30.),np.full(12,-1.)])
        got=d.limited(requested,velocity)
        np.testing.assert_array_equal(got[0],np.full(12,23.4))
        np.testing.assert_array_equal(got[1],np.full(12,20.2))
        np.testing.assert_array_equal(got[2],np.zeros(12))
        np.testing.assert_array_equal(got[3],np.full(12,23.4))

    def test_missing_values_stay_missing(self):
        self.assertIsNone(d.median([None,None]))
        self.assertIsNone(d.rms(np.array([])))
        self.assertEqual(d.median([None,1.,3.]),2.)


if __name__=='__main__':unittest.main()
