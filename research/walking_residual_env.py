"""Gymnasium residual control over the frozen, source-calibrated Go2 walker."""
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import torch

from walking_baseline import Actor, JOINTS, LEGS, DT, CONTROL, TICKS, rotation, torque_limit
from walking_calibrated import calibrated_model
from walking_navigation_hold import Navigator


CONDITIONS = ['healthy', *[leg+'_calf' for leg in LEGS]]
RESIDUAL_SCALE = .10
OBSERVATION_FIELDS = {
    'body_linear_velocity': [0, 3], 'body_angular_velocity': [3, 6],
    'projected_gravity': [6, 9], 'command': [9, 12],
    'joint_position_offset': [12, 24], 'joint_velocity_scaled_005': [24, 36],
    'nominal_target_offset': [36, 48], 'previous_residual_action': [48, 60],
}
REWARD_WEIGHTS = {'velocity_tracking': 1., 'yaw_tracking': .25, 'goal_progress': 10.,
    'height_error': -10., 'tilt_squared': -2., 'cross_track_excess': -5.,
    'stable_goal': .5, 'residual_effort': -.05, 'residual_change': -.01,
    'fall': -10., 'mission_success': 5.}


class WalkingResidualEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, condition='mixed', record_trace=False):
        if condition not in [*CONDITIONS, 'mixed']:
            raise ValueError('Unknown condition')
        self.condition_mode, self.record_trace = condition, bool(record_trace)
        self.render_mode = None
        torch.set_num_threads(1)
        self.actor = Actor('go2_rl_lab')
        for parameter in self.actor.policy.parameters():
            parameter.requires_grad_(False)
        self.model = calibrated_model()
        self.data = mujoco.MjData(self.model)
        assert self.model.opt.timestep == DT and self.model.nu == 12
        assert [self.model.actuator(i).name for i in range(12)] == JOINTS
        np.testing.assert_array_equal(self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]], np.arange(7, 19))
        np.testing.assert_array_equal(self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]], np.arange(6, 18))
        self.feet = [self.model.geom(leg).id for leg in LEGS]
        self.joint_ranges = self.model.jnt_range[self.model.actuator_trnid[:, 0]].copy()
        self.default = np.empty(12)
        self.default[self.actor.order] = self.actor.default
        self.action_space = spaces.Box(-1., 1., (12,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (60,), np.float32)
        self.dt = DT*CONTROL
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        condition = (options or {}).get('condition', self.condition_mode)
        if condition not in [*CONDITIONS, 'mixed']:
            raise ValueError('Unknown reset condition')
        self.condition = str(self.np_random.choice(CONDITIONS)) if condition == 'mixed' else condition
        self.weak = None if self.condition == 'healthy' else self.model.actuator(self.condition).id
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('home').id)
        self.actor.reset()
        self.data.qpos[7:] = self.default+self.np_random.uniform(-.01, .01, 12)
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[2] += .002-min(self.data.geom_xpos[g, 2]-self.model.geom_size[g, 0] for g in self.feet)
        mujoco.mj_forward(self.model, self.data)
        self.origin = self.data.qpos[:2].copy()
        r = rotation(self.data.qpos[3:7])
        self.initial_yaw = float(np.arctan2(r[1, 0], r[0, 0]))
        self.forward = np.array([np.cos(self.initial_yaw), np.sin(self.initial_yaw)])
        self.lateral = np.array([-self.forward[1], self.forward[0]])
        self.goal = self.origin+2*self.forward
        self.navigator = Navigator()
        self.tick, self.onset = 0, None
        self.previous = np.zeros(12)
        self.kp, self.kd = np.full(12, 25.), np.full(12, .5)
        self.max_cross, self.max_tilt = 0., 0.
        self.stable_start, self.hold = None, 0.
        self.ever_goal_entered = self.fallen = False
        self.fault_before_goal = None
        self._done = False
        self._record = {} if self.record_trace else None
        self._append('qpos', self.data.qpos)
        self._append('qvel', self.data.qvel)
        self._update_metrics()
        self._prepare_control()
        return self._observation(), self._info()

    def _append(self, name, value):
        if self._record is not None:
            self._record.setdefault(name, []).append(np.array(value, copy=True))

    def trace(self):
        if self._record is None:
            raise RuntimeError('Construct with record_trace=True to export a trace')
        return {key: np.asarray(value) for key, value in self._record.items()}

    def _prepare_control(self):
        self.command = self.navigator(self.data.qpos, self.origin, self.initial_yaw, self.tick)
        self.nominal_observation = self.actor.observation(self.data.qpos, self.data.qvel, self.command)
        if self.tick == 0:
            self.raw_action = self.nominal_action = np.zeros(12, dtype=np.float32)
            self.nominal_target = self.default.copy()
        else:
            self.raw_action, self.nominal_action = self.actor.predict(self.nominal_observation)
            self.nominal_target = np.empty(12)
            self.nominal_target[self.actor.order] = self.actor.default+.25*self.nominal_action

    def _observation(self):
        if not self.metrics['finite']:
            return np.zeros(60, dtype=np.float32)
        r = rotation(self.data.qpos[3:7])
        # Ordinary motion/command features only; condition, strength, clock and goal position are absent.
        obs = np.r_[r.T@self.data.qvel[:3], self.data.qvel[3:6], r.T@np.array([0., 0., -1.]),
            self.command, self.data.qpos[7:]-self.default, self.data.qvel[6:]*.05,
            self.nominal_target-self.default, self.previous].astype(np.float32)
        if not np.isfinite(obs).all():
            raise RuntimeError('Nonfinite residual observation')
        return obs

    def _update_metrics(self):
        q, v = self.data.qpos, self.data.qvel
        finite = bool(np.isfinite(q).all() and np.isfinite(v).all())
        if not finite:
            self.fallen = True
            self.metrics = {'finite': False, 'stable': False}
            self.hold = 0.
            return
        r = rotation(q[3:7])
        yaw = np.arctan2(r[1, 0], r[0, 0])
        heading = float(abs(np.arctan2(np.sin(yaw-self.initial_yaw), np.cos(yaw-self.initial_yaw))))
        tilt = float(np.rad2deg(np.arccos(np.clip(r[2, 2], -1, 1))))
        error, speed = float(np.linalg.norm(q[:2]-self.goal)), float(np.linalg.norm(v[:2]))
        cross = float(abs((q[:2]-self.origin)@self.lateral))
        stable = error <= .15 and heading <= np.deg2rad(15) and speed <= .05
        self.max_cross, self.max_tilt = max(self.max_cross, cross), max(self.max_tilt, tilt)
        self.fallen |= bool(q[2] < .16 or tilt > 45)
        self.ever_goal_entered |= error <= .15
        if stable:
            if self.stable_start is None:
                self.stable_start = self.tick
            self.hold = (self.tick-self.stable_start)*DT
        else:
            self.stable_start, self.hold = None, 0.
        self.metrics = {'finite': True, 'stable': stable, 'goal_error': error,
            'heading_rad': heading, 'tilt_deg': tilt, 'cross_track': cross,
            'speed': speed, 'height': float(q[2]), 'progress': float((q[:2]-self.origin)@self.forward)}

    def mission_result(self):
        if not self.metrics['finite']:
            return {'success': False, 'fallen': True, 'reason': 'nonfinite state',
                'simulated_s': self.tick*DT, 'fault_onset_tick': self.onset}
        m = self.metrics
        checks = {'full_duration': self.tick == TICKS, 'no_fall': not self.fallen,
            'goal_error': m['goal_error'] <= .15, 'heading_error': bool(m['heading_rad'] <= np.deg2rad(15)),
            'cross_track': self.max_cross <= .20, 'stable_stop': self.hold >= 2.,
            'fault_received': self.weak is None or self.onset is not None}
        if self.weak is not None and self.onset is not None:
            checks['fault_before_goal'] = self.fault_before_goal
        return {'success': bool(all(checks.values())), 'checks': checks, 'fallen': bool(self.fallen),
            'simulated_s': self.tick*DT, 'final_goal_error_m': m['goal_error'],
            'max_cross_track_m': self.max_cross, 'final_heading_error_deg': float(np.rad2deg(m['heading_rad'])),
            'max_tilt_deg': self.max_tilt, 'final_speed_m_s': m['speed'],
            'final_stable_hold_s': self.hold, 'fault_onset_tick': self.onset, 'final_progress_m': m['progress']}

    def _info(self):
        info = {'physics_tick': self.tick, 'time_s': self.tick*DT,
            'fallen': bool(self.fallen), 'fault_onset_tick': self.onset,
            'is_success': bool(self._done and self.mission_result()['success'])}
        if self._done:
            info['episode_result'] = self.mission_result()
        return info

    def step(self, action):
        if self._done:
            raise RuntimeError('Call reset before stepping a finished environment')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('Residual action must contain 12 finite values')
        action = np.clip(action, -1., 1.)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            self._update_metrics()
            self._done = True
            info = self._info()
            info['reward_terms'] = {'fall': -10.}
            return self._observation(), -10., True, False, info
        error_before = self.metrics['goal_error']
        previous = self.previous.copy()
        target = np.clip(self.nominal_target+RESIDUAL_SCALE*action, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        for key, value in [('control_observations', self.nominal_observation),
                ('control_raw_actions', self.raw_action), ('control_actions', self.nominal_action),
                ('control_residual_actions', action)]:
            self._append(key, value)
        for _ in range(CONTROL):
            if self.weak is not None and self.onset is None and (self.data.qpos[:2]-self.origin)@self.forward >= .5:
                self.onset = self.tick
                self.fault_before_goal = not self.ever_goal_entered
            requested = self.kp*(target-self.data.qpos[7:])-self.kd*self.data.qvel[6:]
            limited = torque_limit(requested, self.data.qvel[6:], 'go2_rl_lab', self.model.actuator_ctrlrange)
            applied = limited.copy()
            if self.onset is not None:
                applied[self.weak] *= .1
            for key, value in [('requested_torque', requested), ('limited_torque', limited),
                    ('applied_torque', applied), ('strength', .1 if self.onset is not None else 1.),
                    ('command', self.command), ('target', target), ('kp', self.kp), ('kd', self.kd)]:
                self._append(key, value)
            self.data.ctrl[:] = applied
            mujoco.mj_step(self.model, self.data)
            self.tick += 1
            self._append('actuator_force', self.data.actuator_force)
            self._append('qpos', self.data.qpos)
            self._append('qvel', self.data.qvel)
            if self.record_trace:
                contacts = np.zeros(4, dtype=bool)
                for contact in self.data.contact:
                    for i, geom in enumerate(self.feet):
                        if geom in (contact.geom1, contact.geom2):
                            contacts[i] = True
                self._append('contacts', contacts)
            self._update_metrics()
            if self.fallen or self.tick >= TICKS:
                break
        self.previous = action.copy()
        terminated, truncated = bool(self.fallen), bool(self.tick >= TICKS)
        self._done = terminated or truncated
        if self.metrics['finite']:
            r = rotation(self.data.qpos[3:7])
            body_velocity = r.T@self.data.qvel[:3]
            values = {'velocity_tracking': float(np.exp(-np.sum((body_velocity[:2]-self.command[:2])**2)/.15**2)),
                'yaw_tracking': float(np.exp(-((self.data.qvel[5]-self.command[2])/.5)**2)),
                'goal_progress': error_before-self.metrics['goal_error'],
                'height_error': (self.metrics['height']-.27)**2,
                'tilt_squared': np.deg2rad(self.metrics['tilt_deg'])**2,
                'cross_track_excess': max(0., self.metrics['cross_track']-.20)**2,
                'stable_goal': float(self.metrics['stable']), 'residual_effort': float(np.mean(action**2)),
                'residual_change': float(np.mean((action-previous)**2)),
                'fall': float(terminated), 'mission_success': float(self._done and self.mission_result()['success'])}
            terms = {key: float(REWARD_WEIGHTS[key]*value) for key, value in values.items()}
        else:
            terms = {'fall': -10.}
        reward = float(sum(terms.values()))
        if not np.isfinite(reward):
            raise RuntimeError('Nonfinite reward')
        if not self._done:
            self._prepare_control()
        info = self._info()
        info['reward_terms'] = terms
        return self._observation(), reward, terminated, truncated, info

    def close(self):
        self._done = True
