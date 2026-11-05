import copy
import unittest
import numpy as np
import mujoco
from stable_baselines3.common.env_checker import check_env
import walking_residual_privileged_env as w
import walking_residual_low_noise_pilot as previous


class PrivilegedTests(unittest.TestCase):
    def test_input_parameter_count_and_sigma(self):
        from stable_baselines3 import PPO
        a=w.WalkingResidualEnv('healthy');b=w.OriginalEnv('healthy')
        self.addCleanup(a.close);self.addCleanup(b.close)
        args=dict(seed=0,device='cpu',n_steps=16,batch_size=16,
                  policy_kwargs={'net_arch':[64,64],'log_std_init':float(np.log(.2))})
        new=PPO('MlpPolicy',a,**args);old=PPO('MlpPolicy',b,**args)
        self.assertEqual(sum(p.numel() for p in new.policy.parameters())-sum(p.numel() for p in old.policy.parameters()),1536)
        np.testing.assert_allclose(new.policy.log_std.detach().exp().numpy(),.2,rtol=0,atol=2e-8)
        self.assertTrue(new.policy.log_std.requires_grad)

    def test_check_env(self):
        env=w.WalkingResidualEnv('healthy');self.addCleanup(env.close)
        check_env(env,warn=True)

    def test_no_future_fault_and_exact_parent_dynamics_with_residual(self):
        for condition in w.CONDITIONS:
            with self.subTest(condition=condition):
                env=w.WalkingResidualEnv(condition,record_trace=True)
                base=w.OriginalEnv(condition,record_trace=True)
                self.addCleanup(env.close);self.addCleanup(base.close)
                obs,_=env.reset(seed=23000);original,_=base.reset(seed=23000)
                self.assertEqual(obs.shape,(72,));np.testing.assert_array_equal(obs[:60],original)
                np.testing.assert_array_equal(obs[60:],np.ones(12))
                for item in [env,base]:
                    item.data.qpos[:2]=item.origin+.51*item.forward
                    mujoco.mj_forward(item.model,item.data)
                # Progress has crossed the threshold but no pre-physics latch has run yet.
                np.testing.assert_array_equal(env._observation()[60:],np.ones(12))
                action=np.linspace(-.8,.8,12)
                for _ in range(2):
                    left=env.step(action);right=base.step(action)
                    np.testing.assert_array_equal(left[0][:60],right[0])
                    self.assertEqual(left[1:],right[1:])
                    expected=np.ones(12,dtype=np.float32)
                    if condition!='healthy':expected[env.weak]=.1
                    np.testing.assert_array_equal(left[0][60:],expected)
                for key,value in env.trace().items():np.testing.assert_array_equal(value,base.trace()[key])
                reset,_=env.reset(seed=23001)
                np.testing.assert_array_equal(reset[60:],np.ones(12))

    def test_protocol_and_gate(self):
        import walking_residual_privileged_pilot as p
        old=previous.s.read(previous.OUT/'protocol.json')
        new=copy.deepcopy(old);new['observation_shape']=[72]
        new['observation_fields']=w.OBSERVATION_FIELDS
        p.check_protocol(new,old)
        new['hyperparameters']['gamma']=.98
        with self.assertRaises(AssertionError):p.check_protocol(new,old)
        rows=[dict(condition=c,seed=seed,success=c=='healthy' or seed<23003,fallen=False,interrupted=False,residual_effort=.1)
              for c in w.CONDITIONS for seed in range(23000,23005)]
        self.assertTrue(p.promotion(rows))
        rows[0]['fallen']=True;self.assertFalse(p.promotion(rows))


if __name__=='__main__':unittest.main()
