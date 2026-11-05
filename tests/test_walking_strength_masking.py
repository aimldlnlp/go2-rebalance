import unittest
import numpy as np
import walking_strength_masking as w


class MaskingTests(unittest.TestCase):
    def test_metrics_float64_and_zero_denominator(self):
        rng = np.random.default_rng(4)
        a = dict(ticks=np.array([0, 10]))
        for stage in w.STAGES+['unclipped_target']:
            shape = (2, 12) if stage in w.STAGES[:4]+['unclipped_target'] else (17, 12)
            a[stage+'_f'] = rng.normal(size=shape).astype(np.float32)
            a[stage+'_m'] = a[stage+'_f'].copy()
            if stage != 'raw':
                a[stage+'_m'] += .0123
        value = w.metrics(a, 0, 17, 11)
        self.assertIsNone(value['action_retention']['value'])
        delta = a['action_m'].astype(float)-a['action_f'].astype(float)
        self.assertAlmostEqual(value['action']['rms'], np.linalg.norm(delta)/np.sqrt(delta.size), places=14)

    def test_copy_and_previous_residual(self):
        obs = np.arange(72, dtype=np.float32)/100
        before = obs.copy()
        changed = w.mask(obs)
        np.testing.assert_array_equal(obs, before)
        np.testing.assert_array_equal(changed[:60], obs[:60])
        np.testing.assert_array_equal(changed[60:], np.ones(12))
        self.assertFalse(np.shares_memory(obs, changed))

    def test_latch_boundary_and_windows(self):
        ticks = np.array([0, 10, 20, 30])
        np.testing.assert_array_equal(w.eligibility(ticks, 10), [False, False, True, True])
        np.testing.assert_array_equal(w.eligibility(ticks, 11), [False, False, True, True])
        self.assertFalse(w.eligibility(ticks, None).any())
        self.assertEqual(w.bounds(20, 27, 'first'), (20, 27))
        self.assertIsNone(w.bounds(20, 269, '0.5s'))
        self.assertEqual(w.bounds(20, 270, '0.5s'), (20, 270))
        self.assertIsNone(w.bounds(20, 1019, '2s'))

    def test_clipping_physical_strength_and_partial_interval(self):
        ranges = np.tile([-1., 1.], (12, 1))
        raw = np.full((2, 12), 2., dtype=np.float32)
        action, offset, target = w.action_path(raw, np.full((2, 12), .95), ranges)
        np.testing.assert_array_equal(action, np.ones((2, 12)))
        np.testing.assert_allclose(offset, .1)
        np.testing.assert_array_equal(target, np.ones((2, 12)))
        a = dict(qpos=np.zeros((18, 19)), qvel=np.zeros((18, 18)),
                 kp=np.full((17, 12), 100.), kd=np.ones((17, 12)),
                 strength=np.r_[np.ones(3), np.full(14, .1)])
        limits = np.tile([-23.7, 23.7], (12, 1))
        requested, limited, applied = w.physical_path(target, a, limits, 11)
        self.assertEqual(applied.shape, (17, 12))
        self.assertTrue(np.all(requested != limited))
        np.testing.assert_array_equal(applied[:, :11], limited[:, :11])
        np.testing.assert_array_equal(applied[:3, 11], limited[:3, 11])
        np.testing.assert_array_equal(applied[3:, 11], .1*limited[3:, 11])
        _, _, masked = w.physical_path(target*.5, a, limits, 11)
        np.testing.assert_array_equal(masked[3:, 11], .1*limited[3:, 11])


if __name__ == '__main__':
    unittest.main()
