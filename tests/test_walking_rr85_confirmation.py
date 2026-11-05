import unittest
import numpy as np
import walking_rr85_confirmation as r


class ConfirmationTests(unittest.TestCase):
    def test_fixed_pair_order_and_gate(self):
        cases=r.order()
        self.assertEqual(len(cases),20)
        self.assertEqual(cases[0]['condition'],'healthy')
        self.assertEqual(cases[1]['condition'],'RR_calf')
        self.assertEqual(cases[0]['seed'],cases[1]['seed'])
        self.assertFalse(set(r.SEEDS).intersection(range(23100,23120)))
        rows=[dict(**case,success=True,fallen=False,interrupted=False) for case in cases]
        self.assertTrue(r.gate(rows))
        for field in ['fallen','interrupted']:
            rows[0][field]=True;self.assertFalse(r.gate(rows));rows[0][field]=False
        rows[0]['success']=False;self.assertFalse(r.gate(rows))
        self.assertFalse(r.gate(rows[1:]))

    def test_reused_step_and_healthy_parity(self):
        r.parent.source_contract()
        a=r.parent.SeverityEnv(.85,condition='healthy')
        b=r.parent.w.WalkingResidualEnv('healthy',record_trace=True)
        self.addCleanup(a.close);self.addCleanup(b.close)
        a.reset(seed=23400);b.reset(seed=23400)
        for _ in range(2):
            left=a.step(np.zeros(12));right=b.step(np.zeros(12))
            np.testing.assert_array_equal(left[0],right[0]);self.assertEqual(left[1:],right[1:])
        for key,value in a.trace().items():np.testing.assert_array_equal(value,b.trace()[key])


if __name__=='__main__':unittest.main()
