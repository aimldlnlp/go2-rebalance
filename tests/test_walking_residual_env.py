import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

import walking_residual_env as w
import sudden as s


class ResidualEnvTests(unittest.TestCase):
    def make(self, condition='healthy', record=False):
        env = w.WalkingResidualEnv(condition, record_trace=record)
        self.addCleanup(env.close)
        env.reset(seed=20500)
        return env

    def test_gymnasium_checker_and_observation_contract(self):
        env = self.make('mixed')
        check_env(env, warn=True)
        obs, info = env.reset(seed=987645)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(obs.shape, (60,))
        self.assertEqual(sum(end-start for start, end in w.OBSERVATION_FIELDS.values()), 60)
        self.assertFalse(any(p.requires_grad for p in env.actor.policy.parameters()))
        self.assertFalse(env.actor.policy.training)

    def test_reset_history_navigator_and_no_privileged_observation(self):
        healthy, fault = self.make(), self.make('FR_calf')
        np.testing.assert_array_equal(healthy._observation(), fault._observation())
        expected = healthy._observation().copy()
        healthy.step(np.ones(12))
        healthy.navigator.stopped = True
        obs, _ = healthy.reset(seed=20500)
        np.testing.assert_array_equal(obs, expected)
        np.testing.assert_array_equal(healthy.actor.previous, np.zeros(12))
        self.assertFalse(healthy.navigator.stopped)
        healthy.data.qpos[:2] += [5., -3.]
        healthy.origin += [5., -3.]
        healthy.goal += [5., -3.]
        np.testing.assert_array_equal(healthy._observation(), expected)

    def test_zero_residual_first_controls_match_archived_trace(self):
        env = self.make(record=True)
        for _ in range(3):
            env.step(np.zeros(12))
        actual = env.trace()
        with np.load(s.ROOT/'results/walking_navigation_hold/diagnosis/healthy_20500.npz') as reference:
            for key in reference.files:
                value = actual[key]
                np.testing.assert_array_equal(value, reference[key][:len(value)], err_msg=key)

    def test_invalid_actions_and_residual_clipping_before_pd(self):
        env = self.make(record=True)
        for action in [np.zeros(11), np.full(12, np.nan), np.full(12, np.inf)]:
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.tick, 0)
        obs, reward, terminated, truncated, info = env.step(np.full(12, 9.))
        a = env.trace()
        np.testing.assert_array_equal(a['control_residual_actions'][0], np.ones(12))
        np.testing.assert_allclose(a['target'][0], np.clip(env.default+.1, env.joint_ranges[:, 0], env.joint_ranges[:, 1]), rtol=0, atol=0)
        np.testing.assert_allclose(a['requested_torque'], 25*(a['target']-a['qpos'][:-1, 7:])-.5*a['qvel'][:-1, 6:], rtol=0, atol=1e-12)
        self.assertTrue(np.all(a['target'] >= env.joint_ranges[:, 0]))
        self.assertTrue(np.all(a['target'] <= env.joint_ranges[:, 1]))
        self.assertTrue(np.all(np.abs(a['limited_torque']) <= np.array([23.7, 23.7, 45.43]*4)))
        self.assertAlmostEqual(reward, sum(info['reward_terms'].values()))
        self.assertLess(info['reward_terms']['residual_effort'], 0)
        self.assertLess(info['reward_terms']['residual_change'], 0)

    def test_fault_latches_first_pre_step_and_scales_only_selected_motor(self):
        env = self.make('FR_calf', record=True)
        real_step = mujoco.mj_step
        calls = 0
        def cross_after_first_step(model, data):
            nonlocal calls
            real_step(model, data)
            calls += 1
            if calls == 1:
                data.qpos[0] = env.origin[0]+.51
        with patch.object(w.mujoco, 'mj_step', cross_after_first_step):
            env.step(np.full(12, .5))
        self.assertEqual(env.onset, 1)
        a = env.trace()
        expected = a['limited_torque'].copy()
        expected[1:, env.weak] *= .1
        np.testing.assert_array_equal(a['applied_torque'], expected)
        np.testing.assert_allclose(a['actuator_force'], expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(a['strength'], np.r_[1., np.full(9, .1)])

    def test_time_limit_and_step_after_done(self):
        env = self.make()
        env.tick = w.TICKS-w.CONTROL
        env._prepare_control()
        obs, reward, terminated, truncated, info = env.step(np.zeros(12))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['time_s'], 20.)
        self.assertTrue(info['episode_result']['checks']['full_duration'])
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(12))

    def test_fall_terminates_at_exact_physics_tick(self):
        env = self.make()
        real_step = mujoco.mj_step
        def collapse(model, data):
            real_step(model, data)
            data.qpos[2] = .15
        with patch.object(w.mujoco, 'mj_step', collapse):
            obs, reward, terminated, truncated, info = env.step(np.zeros(12))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['physics_tick'], 1)
        self.assertFalse(info['episode_result']['success'])
        self.assertEqual(info['reward_terms']['fall'], -10.)

    def test_nonfinite_state_fails_with_finite_api_outputs(self):
        env = self.make()
        env.data.qpos[7] = np.nan
        obs, reward, terminated, truncated, info = env.step(np.zeros(12))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(np.isfinite(obs).all())
        self.assertTrue(np.isfinite(reward))
        self.assertEqual(info['episode_result']['reason'], 'nonfinite state')

    def test_reward_is_translation_invariant_and_uses_goal_progress(self):
        first, shifted = self.make(), self.make()
        shifted.data.qpos[:2] += [3., -2.]
        shifted.origin += [3., -2.]
        shifted.goal += [3., -2.]
        mujoco.mj_forward(shifted.model, shifted.data)
        shifted._update_metrics()
        shifted._prepare_control()
        a, b = first.step(np.zeros(12)), shifted.step(np.zeros(12))
        self.assertAlmostEqual(a[1], b[1], places=8)
        self.assertIn('goal_progress', a[4]['reward_terms'])
        self.assertIn('velocity_tracking', a[4]['reward_terms'])
        self.assertIn('height_error', a[4]['reward_terms'])
        self.assertIn('stable_goal', a[4]['reward_terms'])

    def test_stable_hold_boundary_and_success_bonus(self):
        env = self.make()
        env.data.qpos[0] = env.origin[0]+1.9
        env.tick = 9990
        mujoco.mj_forward(env.model, env.data)
        env._update_metrics()
        env.stable_start = 9000
        env.hold = 1.98
        env._prepare_control()
        env.nominal_target = env.data.qpos[7:].copy()
        obs, reward, terminated, truncated, info = env.step(np.zeros(12))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['episode_result']['final_stable_hold_s'], 2.)
        self.assertTrue(info['is_success'])
        self.assertEqual(info['reward_terms']['stable_goal'], .5)
        self.assertEqual(info['reward_terms']['mission_success'], 5.)
        env.hold = 1.998
        self.assertFalse(env.mission_result()['success'])


if __name__ == '__main__':
    unittest.main()
