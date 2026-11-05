import unittest
import numpy as np
import walking_strength_masking_rollout as study


class Tests(unittest.TestCase):
    def test_only_observation_override(self):
        self.assertIs(study.MaskedEnv.step,study.original.WalkingResidualEnv.step)
        self.assertIs(study.MaskedEnv.reset,study.original.WalkingResidualEnv.reset)

    def test_dynamics_latch_and_applied_history(self):
        factual = study.original.WalkingResidualEnv('RR_calf',record_trace=True)
        masked = study.MaskedEnv('RR_calf',record_trace=True)
        try:
            f,_ = factual.reset(seed=23000)
            m,_ = masked.reset(seed=23000)
            np.testing.assert_array_equal(f,m)
            # Move both fixtures just past the trigger; suffix changes only after the step latches it.
            for env in [factual,masked]:
                env.data.qpos[0] = .501
            self.assertTrue(np.all(factual._observation()[60:]==1))
            action = np.linspace(-2,2,12)
            f,fr,ft,fu,_ = factual.step(action)
            m,mr,mt,mu,_ = masked.step(action)
            self.assertEqual(factual.onset,masked.onset)
            self.assertIsNotNone(factual.onset)
            self.assertEqual((fr,ft,fu),(mr,mt,mu))
            np.testing.assert_array_equal(f[:60],m[:60])
            np.testing.assert_array_equal(m[60:],np.ones(12))
            self.assertEqual(f[71],np.float32(.1))
            np.testing.assert_array_equal(m[48:60],np.clip(action,-1,1).astype(np.float32))
            for key in factual.trace():
                np.testing.assert_array_equal(factual.trace()[key],masked.trace()[key])
            trace = masked.trace()
            np.testing.assert_array_equal(trace['applied_torque'][:,11],.1*trace['limited_torque'][:,11])
            f,_ = factual.reset(seed=23001,options={'condition':'healthy'})
            m,_ = masked.reset(seed=23001,options={'condition':'healthy'})
            np.testing.assert_array_equal(f,m)
        finally:
            factual.close()
            masked.close()


if __name__ == '__main__':
    unittest.main()
