import math
import unittest

import numpy as np

import analyze_walking_next_step as a


class DecisionTests(unittest.TestCase):
    def test_missing_fallen_episode_is_not_scored(self):
        labels=np.tile(np.arange(5),5)[:-1]
        seeds=np.repeat(np.arange(5),5)[:-1]
        result=a.cohort_probe(np.zeros((24,3)),labels,seeds)
        self.assertEqual(result['observed_episodes'],24)
        self.assertIsNone(result['accuracy'])
        self.assertIsNone(result['correct'])

    def test_clipped_normal_known_integral_and_low_noise_limit(self):
        clip,second=a.clipped_normal(np.zeros((1,1)),np.ones((1,1)))
        self.assertAlmostEqual(float(clip[0,0]),.31731050786291415)
        self.assertAlmostEqual(float(second[0,0]),.5160585509617133)
        clip,second=a.clipped_normal(np.array([0.]),np.array([.01]))
        self.assertAlmostEqual(float(clip[0]),0.)
        self.assertAlmostEqual(float(second[0]),.0001)

    def test_identical_inputs_do_not_reveal_future_condition(self):
        labels=np.tile(np.arange(5),5);seeds=np.repeat(np.arange(5),5)
        x=np.zeros((25,3))
        for seed in range(5):x[seeds==seed]=seed*.01
        result=a.centroid_probe(x,labels,seeds)
        self.assertEqual(result['correct'],5)

    def test_leave_seed_out_separable_features(self):
        labels=np.tile(np.arange(5),5);seeds=np.repeat(np.arange(5),5)
        x=np.eye(5)[labels]+seeds[:,None]*.001
        result=a.centroid_probe(x,labels,seeds)
        self.assertEqual(result['correct'],25)


if __name__=='__main__':unittest.main()
