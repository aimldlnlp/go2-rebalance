import copy
import unittest

import numpy as np
from stable_baselines3 import PPO

import walking_residual_low_noise_pilot as p


class LowNoiseTests(unittest.TestCase):
    def test_only_initial_log_std_changes_and_remains_trainable(self):
        env=p.w.WalkingResidualEnv('healthy')
        self.addCleanup(env.close)
        hp=p.s.read(p.parent.OUT/'protocol.json')['hyperparameters']
        with p.preserved_rng():
            previous=PPO('MlpPolicy',env,seed=0,device='cpu',
                policy_kwargs={'net_arch':[64,64],'log_std_init':0.})
            revised=PPO('MlpPolicy',env,seed=0,device='cpu',
                **{key:value for key,value in hp.items() if key!='net_arch'},
                policy_kwargs={'net_arch':[64,64],'log_std_init':p.LOG_STD_INIT})
        old,new=p.policy_record(previous),p.policy_record(revised)
        self.assertEqual(old['shared_non_log_std_parameters_sha256'],new['shared_non_log_std_parameters_sha256'])
        np.testing.assert_array_equal(old['sigma'],np.ones(12))
        np.testing.assert_allclose(new['sigma'],.2,rtol=0,atol=2e-8)
        self.assertTrue(revised.policy.log_std.requires_grad)
        for name,value in previous.policy.state_dict().items():
            if name!='log_std':
                np.testing.assert_array_equal(value.detach().numpy(),revised.policy.state_dict()[name].detach().numpy())

    def test_protocol_rejects_every_other_configuration_change(self):
        old=p.s.read(p.parent.OUT/'protocol.json')
        revised=copy.deepcopy(old)
        revised['ppo_fixed_defaults']['log_std_init']=p.LOG_STD_INIT
        p.check_protocol(revised,old)
        for field,key,value in [('hyperparameters','gamma',.98),('ppo_fixed_defaults','ent_coef',.01),
                ('reward_weights','goal_progress',20.)]:
            broken=copy.deepcopy(revised);broken[field][key]=value
            with self.assertRaises(AssertionError):p.check_protocol(broken,old)
        with self.assertRaises(AssertionError):p.check_protocol(old,old)

    def test_gate_preserves_all_fault_and_safety_requirements(self):
        rows=[{'condition':c,'seed':seed,'success':c=='healthy' or seed<23003,
            'fallen':False,'interrupted':False,'residual_effort':.1}
            for c in p.w.CONDITIONS for seed in range(23000,23005)]
        self.assertTrue(p.promotion(rows))
        for field in ['fallen','interrupted']:
            rows[0][field]=True
            self.assertFalse(p.promotion(rows));rows[0][field]=False
        rows[-5]['success']=False
        self.assertFalse(p.promotion(rows))


if __name__=='__main__':unittest.main()
