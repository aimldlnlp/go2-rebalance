import unittest

import numpy as np

import walking_noise_ablation as a


class NoiseTests(unittest.TestCase):
    def test_distribution_mean_matches_frozen_deterministic_policy(self):
        model=a.frozen_policy(a.pilot.OUT/'checkpoint_131072.zip')
        with np.load(a.pilot.OUT/'validation_131072/healthy_23000.npz',allow_pickle=False) as trace:
            for index in [0,100,500,999]:
                obs=trace['residual_observations'][index]
                mean,sigma=a.parameters(model,obs)
                _,action=a.sample_action(mean,sigma,np.zeros(12,dtype=np.float32),0.)
                np.testing.assert_array_equal(action,model.predict(obs,deterministic=True)[0])
                np.testing.assert_array_equal(action,trace['residual_policy_actions'][index])

    def test_sampling_formula_and_clipping(self):
        mean=np.linspace(-.4,.4,12,dtype=np.float32);sigma=np.full(12,.97,dtype=np.float32)
        epsilon=np.linspace(-2,2,12,dtype=np.float32)
        for scale in [0.,.2,1.]:
            raw,chosen=a.sample_action(mean,sigma,epsilon,scale)
            expected=mean if scale==0 else mean+(sigma*np.float32(scale))*epsilon
            np.testing.assert_array_equal(raw,expected)
            np.testing.assert_array_equal(chosen,np.clip(expected,-1,1))
        np.testing.assert_array_equal(a.sample_action(mean,sigma,epsilon,0.)[1],mean)

    def test_invalid_noise_or_sigma_rejected(self):
        good=np.ones(12,dtype=np.float32)
        for mean,sigma,epsilon,scale in [(good,good,np.zeros(11),1.),(good,-good,good,.2),
            (good,good,np.full(12,np.nan),1.),(good,good,good,.5)]:
            with self.assertRaises(ValueError):a.sample_action(mean,sigma,epsilon,scale)

    def test_common_noise_is_not_rescaled_or_mutated(self):
        epsilon=np.random.default_rng(33000).standard_normal((1000,12)).astype(np.float32)
        copy=epsilon.copy();mean=np.zeros(12,dtype=np.float32);sigma=np.ones(12,dtype=np.float32)
        low,_=a.sample_action(mean,sigma,epsilon[0],.2)
        full,_=a.sample_action(mean,sigma,epsilon[0],1.)
        np.testing.assert_array_equal(low,np.float32(.2)*full)
        np.testing.assert_array_equal(epsilon,copy)


if __name__=='__main__':unittest.main()
