import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import walking_baseline as w


class WalkingTests(unittest.TestCase):
    def test_upstream_gravity_and_motor_envelope_match(self):
        source=Path('vendor/walking_baselines/go2_rl_lab/source_mujoco.py').read_text()
        selected=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ['get_gravity_orientation','clip_torque_speed']]
        scope={'np':np}
        exec(compile(ast.Module(body=selected,type_ignores=[]),'<reviewed pure upstream helpers>','exec'),scope)
        rng=np.random.default_rng(987651)
        for _ in range(20):
            q=rng.normal(size=4);q/=np.linalg.norm(q)
            np.testing.assert_allclose(w.rotation(q).T@np.array([0,0,-1]),scope['get_gravity_orientation'](q),atol=1e-7,rtol=0)
            tau=rng.uniform(-60,60,12);velocity=rng.uniform(-40,40,12)
            expected=scope['clip_torque_speed'](tau,velocity,{'use_torque_speed_curve':True,'X1':13.5,'X2':30,'Y1':20.2,'Y2':23.4})
            np.testing.assert_allclose(w.torque_limit(tau,velocity,'go2_rl_lab',np.tile([-100,100],(12,1))),expected,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(w.torque_limit(np.full(12,99.),np.zeros(12),'genesis',np.tile([-23.7,23.7],(12,1))),np.full(12,23.7))

    def test_policy_mapping_cpu_shape_and_history_reset(self):
        torch.set_num_threads(1)
        for candidate in w.CANDIDATES:
            actor=w.Actor(candidate)
            np.testing.assert_array_equal(actor.order,np.arange(12) if candidate=='go2_rl_lab' else [3,4,5,0,1,2,9,10,11,6,7,8])
            q=np.r_[0,0,.3,1,0,0,0,np.zeros(12)]
            q[7+actor.order]=actor.default
            obs=actor.observation(q,np.zeros(18),np.zeros(3))
            self.assertEqual(obs.shape,(45 if candidate=='go2_rl_lab' else 49,))
            np.testing.assert_array_equal(obs[3:6],[0,0,-1])
            first=actor.predict(obs)[0]
            actor.predict(obs)
            actor.reset()
            np.testing.assert_array_equal(first,actor.predict(obs)[0])
            self.assertTrue(np.isfinite(first).all())
            with self.assertRaises(ValueError):actor.predict(np.full_like(obs,np.nan))

    def test_navigation_is_bounded_and_stops_in_goal(self):
        q=np.r_[0,0,.3,1,0,0,0,np.zeros(12)]
        np.testing.assert_array_equal(w.mission_command(q,np.zeros(2),0,900),np.zeros(3))
        for tick in [1000,1010,1100,1500]:
            cmd=w.mission_command(q,np.zeros(2),0,tick)
            self.assertLessEqual(np.linalg.norm(cmd[:2]),.2+1e-12)
        q[0]=1.95
        np.testing.assert_array_equal(w.mission_command(q,np.zeros(2),0,1500),np.zeros(3))

    def test_mission_boundaries_and_nonfinite_fail(self):
        q=np.zeros((10001,19));q[:,2]=.3;q[:,3]=1
        q[:,0]=np.minimum(np.arange(10001)/4000,2)
        v=np.zeros((10001,18));a={'qpos':q,'qvel':v}
        self.assertTrue(w.summarize(a,'healthy',np.zeros(2),0,None)['success'])
        self.assertFalse(w.summarize(a,'FL_calf',np.zeros(2),0,None)['success'])
        v[-1,0]=.050000001
        self.assertFalse(w.summarize(a,'healthy',np.zeros(2),0,None)['checks']['stable_stop'])
        v[-1,0]=0;q[3000,1]=.200000001
        self.assertFalse(w.summarize(a,'healthy',np.zeros(2),0,None)['checks']['cross_track'])
        q[3000,1]=0;q[-1,2]=.159
        self.assertFalse(w.summarize(a,'healthy',np.zeros(2),0,None)['success'])
        q[-1,2]=np.nan
        self.assertFalse(w.summarize(a,'healthy',np.zeros(2),0,None)['success'])

    def test_one_step_fault_scales_only_selected_torque(self):
        actor=w.Actor('go2_rl_lab')
        original=w.mujoco.mj_step
        def move_after_first(model,data):
            original(model,data)
            if data.time<.003:data.qpos[0]=.51
        with patch.object(w,'TICKS',4),patch.object(w.mujoco,'mj_step',move_after_first):
            raw,row=w.episode(actor,987650,'FR_calf')
        self.assertEqual(row['fault_onset_tick'],1)
        expected=raw['limited_torque'].copy();expected[1:,5]*=.1
        np.testing.assert_array_equal(raw['applied_torque'],expected)
        np.testing.assert_allclose(raw['actuator_force'],raw['applied_torque'],atol=1e-12,rtol=0)
        self.assertEqual(raw['qpos'].shape,(5,19))
        self.assertEqual(raw['control_observations'].shape,(1,45))

    def test_refuses_overwrite_and_healthy_gate(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(w,'OUT',Path(folder)):
            with self.assertRaisesRegex(AssertionError,'overwrite'):w.prepare()
            (Path(folder)/'execution.json').write_text('{}')
            with self.assertRaisesRegex(AssertionError,'rerun'):w.run()


if __name__=='__main__':unittest.main()
