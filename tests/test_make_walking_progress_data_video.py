import unittest
import numpy as np
import make_walking_progress_data_video as v


class Tests(unittest.TestCase):
    def test_timing_and_bin_boundaries(self):
        dummy={'cases':{n:{23000:{'row':{'fault_onset_tick':21}}} for n in v.FOLDERS}}
        self.assertEqual(v.ticks('comparison',499,dummy)['original'],9980)
        self.assertEqual(v.ticks('detail',0,dummy)['original'],30)
        self.assertEqual(v.ticks('detail',249,dummy)['original'],2520)
        low=np.zeros((151,12));high=np.full((151,12),5.)
        b=v.bin_span(low,high)
        np.testing.assert_array_equal(b[:,:2],np.full((12,2),5.))
        self.assertTrue(np.isnan(b[:,2:]).all())

    def test_data_integrity_without_simulation_or_policy(self):
        with v.forbid():
            data=v.load()
        self.assertEqual(data['totals'],{'original':{'success':0,'falls':7},'progress':{'success':0,'falls':6}})
        for group in data['cases'].values():
            self.assertEqual(len(group),5)
            for case in group.values():
                q=case['qpos'];r=case['row']
                goal=np.asarray(r['initial_xy'])+2*np.array([np.cos(r['initial_yaw']),np.sin(r['initial_yaw'])])
                np.testing.assert_allclose(case['error'],np.sqrt(np.sum((q[:,:2]-goal)**2,axis=1)),atol=1e-14,rtol=0)
        for name in v.FOLDERS:
            e=data['envelopes'][name]
            for j in [0,37,99]:
                np.testing.assert_allclose(e['binned'][:,j],(e['high'][j*100:(j+1)*100]-e['low'][j*100:(j+1)*100]).mean(axis=0),atol=1e-14,rtol=0)


if __name__=='__main__':
    unittest.main()
