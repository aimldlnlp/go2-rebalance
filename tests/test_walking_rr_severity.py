import inspect
import textwrap
import unittest
import numpy as np
import mujoco
import review_walking
import walking_rr_severity as r


class SeverityTests(unittest.TestCase):
    def test_step_and_auditor_only_have_declared_changes(self):
        r.source_contract()
        expected=inspect.getsource(review_walking.verify_episode).replace('w.','baseline.')
        expected=expected.replace('applied[onset:,weak]*=.1',"applied[onset:,weak]*=row['strength_remaining']")
        expected=expected.replace('strength[onset:]=.1',"strength[onset:]=row['strength_remaining']")
        self.assertEqual(textwrap.dedent(inspect.getsource(r.verify_episode)),textwrap.dedent(expected))

    def fault_now(self,env):
        env.reset(seed=23000)
        env.data.qpos[:2]=env.origin+.51*env.forward
        mujoco.mj_forward(env.model,env.data)

    def test_ten_percent_matches_parent_after_latch(self):
        a=r.SeverityEnv(.1);b=r.w.WalkingResidualEnv('RR_calf',record_trace=True)
        self.addCleanup(a.close);self.addCleanup(b.close)
        self.fault_now(a);self.fault_now(b)
        for _ in range(2):
            left=a.step(np.zeros(12,dtype=np.float32));right=b.step(np.zeros(12,dtype=np.float32))
            np.testing.assert_array_equal(left[0],right[0]);self.assertEqual(left[1:],right[1:])
        self.assertEqual(a.onset,0)
        for key,value in a.trace().items():np.testing.assert_array_equal(value,b.trace()[key],err_msg=key)

    def test_strength_after_limiter_selected_calf_only(self):
        for strength in r.STRENGTHS:
            with self.subTest(strength=strength):
                env=r.SeverityEnv(strength);self.addCleanup(env.close);self.fault_now(env)
                obs,*_=env.step(np.zeros(12,dtype=np.float32));a=env.trace()
                expected=a['limited_torque'].copy();expected[:,11]*=strength
                np.testing.assert_array_equal(a['applied_torque'],expected)
                np.testing.assert_array_equal(a['strength'],np.full(10,strength))
                np.testing.assert_array_equal(obs[48:60],np.zeros(12))
                np.testing.assert_array_equal(a['control_residual_actions'],np.zeros((1,12)))
        for invalid in [0,1.,.2,float('nan'),float('inf'),None]:
            with self.assertRaises(ValueError):r.SeverityEnv(invalid)

    def test_prefix_ignores_postfault_change_and_detects_prefault_change(self):
        a=r.read_arrays(r.ARCHIVE/'RR_calf_23000.npz');b={k:v.copy() for k,v in a.items()}
        row=r.s.read(r.ARCHIVE/'RR_calf_23000.json');onset=row['fault_onset_tick']
        a['applied_torque'][onset,11]+=1
        r.check_prefix(a,b,onset)
        a['applied_torque'][onset-1,11]+=1
        with self.assertRaises(AssertionError):r.check_prefix(a,b,onset)


if __name__=='__main__':unittest.main()
