import unittest
import compare_walking_low_noise as c


class ComparisonTests(unittest.TestCase):
    def rows(self):
        return [dict(seed=i,success=False,fallen=False,stall_window_start_s=None,
                     **{k:float(i) for k in c.METRICS}) for i in range(5)]

    def test_pairing_and_missing(self):
        a=self.rows();b=self.rows()
        a[0]['fallen']=True;b[1]['fallen']=True
        for r in a+b:r['post_weak_torque_limit_fraction']=None
        s=c.paired_summary(a,b)
        self.assertEqual(s['falls'],[1,1])
        self.assertEqual((s['fall_rescues'],s['fall_harms']),(1,1))
        self.assertIsNone(s['paired_median_changes']['post_weak_torque_limit_fraction'])
        self.assertEqual(s['paired_median_changes']['post_net_progress_m'],0.)
        with self.assertRaises(AssertionError):c.paired_summary(a,b[::-1])


if __name__=='__main__':unittest.main()
