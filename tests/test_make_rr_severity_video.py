import unittest
from unittest.mock import patch
import numpy as np
import mujoco
import make_rr_severity_video as v


class SavedVideoTests(unittest.TestCase):
    def test_chapter_timing_and_saved_sample_indices(self):
        self.assertEqual(sum(s for _,s in v.CHAPTERS),60)
        self.assertEqual(v.sample_tick(0),0)
        self.assertEqual(v.sample_tick(499),9980)
        self.assertEqual(v.sample_tick(249,rate=2),9960)
        self.assertEqual(v.sample_tick(100,start_s=16),10000)
        self.assertEqual(v.sample_tick(199,start_s=16),10000)

    def test_sources_archive_labels_and_aggregate_scope(self):
        with patch.object(mujoco,'mj_step',side_effect=AssertionError('No integration allowed')):
            cases=v.load_cases()
        self.assertEqual(list(cases),v.ORDER)
        self.assertEqual([k for k,c in cases.items() if c['row']['historical_control']],['healthy','rr_10'])
        self.assertTrue(all(c['summary']['episodes']==5 for c in cases.values()))
        self.assertEqual([cases[k]['summary']['successes'] for k in v.ORDER],[5,0,0,0,0])
        self.assertAlmostEqual(cases['rr_75']['first_goal_entry_s'],18.832)
        self.assertAlmostEqual(cases['rr_75']['hold'][-1],.772)
        for case in cases.values():np.testing.assert_array_equal(case['qpos'][0],cases['healthy']['qpos'][0])


if __name__=='__main__':unittest.main()
