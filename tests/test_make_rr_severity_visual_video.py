import unittest
import numpy as np
import make_rr_severity_visual_video as v


class VisualVideoTests(unittest.TestCase):
    def test_sampling_matches_duration_and_is_synchronized(self):
        self.assertEqual(sum(s for _,s in v.CHAPTERS)*v.FPS,1050)
        self.assertEqual(v.frame_tick('paired',499),9980)
        self.assertEqual(v.frame_tick('levels',249),9960)
        self.assertEqual(v.frame_tick('hold',199),9990)
        self.assertEqual(v.frame_tick('results',0),10000)

    def test_indicator_resets_and_fault_boundary(self):
        c={'row':{'fault_onset_tick':2,'strength_remaining':.75},
           'progress':np.arange(5),'hold':np.array([0,1,2,0,.5]),'error':np.array([1,.1,.1,.2,.1])}
        self.assertEqual(v.indicators(c,1)['strength'],1.)
        self.assertEqual(v.indicators(c,2)['strength'],.75)
        self.assertEqual(v.indicators(c,2)['hold_fraction'],1.)
        self.assertEqual(v.indicators(c,3)['hold_fraction'],0.)
        self.assertFalse(v.indicators(c,3)['in_goal'])

    def test_saved_full_resolution_indicators_without_integration(self):
        with v.forbidden():
            cases=v.old.load_cases();proof=v.validate_indicators(cases)
        self.assertEqual(sum(r['states'] for r in proof.values()),50005)
        self.assertAlmostEqual(proof['rr_75']['final_hold_s'],.772)
        self.assertAlmostEqual(proof['rr_75']['first_goal_entry_s'],18.832)


if __name__=='__main__':unittest.main()
