import unittest

import numpy as np

import walking_residual_ablation as a
import walking_residual_env as w


class AblationTests(unittest.TestCase):
    def test_gate_uses_planar_command_only_without_changing_proposal(self):
        obs=np.zeros(60,dtype=np.float32)
        obs[11]=.5
        proposal=np.full(12,.4,dtype=np.float32)
        applied,off=a.gated_action(proposal,obs,'zero_command_off')
        self.assertTrue(off)
        np.testing.assert_array_equal(applied,np.zeros(12))
        np.testing.assert_array_equal(proposal,np.full(12,.4,dtype=np.float32))
        applied,off=a.gated_action(proposal,obs,'always')
        self.assertFalse(off)
        np.testing.assert_array_equal(applied,proposal)
        obs[9]=1e-6
        applied,off=a.gated_action(proposal,obs,'zero_command_off')
        self.assertFalse(off)
        np.testing.assert_array_equal(applied,proposal)

    def test_invalid_inputs_fail(self):
        for action,obs,mode in [(np.zeros(11),np.zeros(60),'always'),
                (np.full(12,np.nan),np.zeros(60),'always'),
                (np.zeros(12),np.full(60,np.inf),'always'),
                (np.zeros(12),np.zeros(60),'unknown')]:
            with self.assertRaises(ValueError):a.gated_action(action,obs,mode)

    def test_reward_reconstruction_agrees_with_actual_env_components(self):
        env=w.WalkingResidualEnv('healthy',record_trace=True)
        self.addCleanup(env.close)
        env.reset(seed=20500)
        rewards,components=[],[]
        for i in range(3):
            _,reward,_,_,info=env.step(np.full(12,.2*i,dtype=np.float32))
            rewards.append(reward)
            components.append([info['reward_terms'][key] for key in a.TERMS])
        trace=env.trace();trace['control_rewards']=np.asarray(rewards)
        row={**env.mission_result(),'initial_xy':env.origin.tolist(),'initial_yaw':env.initial_yaw}
        phases,weighted=a.decompose(trace,row)
        np.testing.assert_allclose(np.column_stack([weighted[key] for key in a.TERMS]),components,rtol=0,atol=1e-10)
        self.assertEqual(sum(r['controls'] for r in phases),3)
        self.assertAlmostEqual(sum(r['duration_s'] for r in phases),.06)
        stop=next(r for r in phases if r['phase']=='stop')
        self.assertEqual(stop['controls'],0)
        self.assertIsNone(stop['residual_rms_rad'])

    def test_historical_reward_and_phase_partition(self):
        folder=a.pilot.OUT/'validation_131072'
        row=a.s.read(folder/'healthy_23001.json')
        with np.load(folder/'healthy_23001.npz',allow_pickle=False) as trace:
            phases,weighted=a.decompose(trace,row)
        self.assertEqual(sum(r['controls'] for r in phases),1000)
        self.assertEqual(next(r['controls'] for r in phases if r['phase']=='settle'),100)
        self.assertEqual(next(r['controls'] for r in phases if r['phase']=='ramp_zero'),1)
        self.assertAlmostEqual(sum(v.sum() for v in weighted.values()),row['episode_return'],places=8)


if __name__=='__main__':unittest.main()
