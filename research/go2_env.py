"""Headless Go2 standing with torque PD and optional joint-target residuals."""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

MODEL = Path(__file__).resolve().parents[1] / "vendor/mujoco_menagerie/unitree_go2/scene.xml"


class Go2Balance(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, strength=1.0, training=False, residual_scale=0.25,
                 residual_penalty=0.03, tilt_deadband_deg=None):
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        if not np.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be finite and positive")
        if not np.isfinite(residual_penalty) or residual_penalty < 0:
            raise ValueError("residual_penalty must be finite and nonnegative")
        if tilt_deadband_deg is not None and not 0 <= tilt_deadband_deg < 45:
            raise ValueError("tilt_deadband_deg must be in [0, 45)")
        self.residual_scale = residual_scale
        self.residual_penalty = residual_penalty
        self.tilt_deadband_deg = tilt_deadband_deg
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        self.base = self.model.body("base").id
        self.weak_motor = self.model.actuator("FL_calf").id
        self.joints = self.model.actuator_trnid[:, 0]
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        self.home = self.model.key("home").qpos[self.qadr].copy()
        self.limits = self.model.actuator_ctrlrange.copy()
        self.strength = strength
        self.training = training
        self.frame_skip = 10
        self.dt = self.model.opt.timestep * self.frame_skip
        self.horizon = round(30 / self.dt)
        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (60,), np.float32)
        self.previous = np.zeros(12)

    def motor_strength(self):
        # Ramp the selected motor only, after five healthy seconds.
        progress = np.clip((self.data.time - 5.0) / 10.0, 0.0, 1.0)
        return float(1.0 - progress * (1.0 - self.strength))

    def _observation(self):
        rotation = self.data.xmat[self.base].reshape(3, 3)
        return np.concatenate((
            rotation.T @ np.array([0.0, 0.0, -1.0]),
            rotation.T @ self.data.qvel[:3], self.data.qvel[3:6],
            self.data.qpos[self.qadr] - self.home,
            self.data.qvel[self.vadr], self.data.qpos[:3], self.previous,
            self.data.qpos[self.qadr] - self.last_q,
        )).astype(np.float32)

    def metrics(self):
        upright = float(self.data.xmat[self.base].reshape(3, 3)[2, 2])
        tilt = float(np.arccos(np.clip(upright, -1, 1)))
        height = float(self.data.qpos[2])
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        fallen = bool(not finite or height < 0.16 or tilt > np.deg2rad(45))
        return {"time_s": float(self.data.time), "tilt_deg": float(np.rad2deg(tilt)),
                "height_m": height, "fallen": fallen, "strength": self.motor_strength()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.data.qpos[self.qadr] += self.np_random.uniform(-0.01, 0.01, 12)
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        self.previous[:] = 0
        self.last_q = self.data.qpos[self.qadr].copy()
        self.steps = 0
        if self.training:
            self.strength = float(self.np_random.choice([1.0, 0.75, 0.5, 0.25, 0.1]))
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), self.metrics()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("action must contain 12 finite values")
        action = np.clip(action, -1.0, 1.0)
        if self.tilt_deadband_deg is not None:
            # Preserve PD below the deadband; blend the learned residual over one degree.
            action *= np.clip(self.metrics()["tilt_deg"] - self.tilt_deadband_deg, 0.0, 1.0)
        self.last_q = self.data.qpos[self.qadr].copy()
        target = np.clip(self.home + self.residual_scale * action,
                         self.model.jnt_range[self.joints, 0],
                         self.model.jnt_range[self.joints, 1])
        failed = False
        for _ in range(self.frame_skip):
            torque = 50.0 * (target - self.data.qpos[self.qadr]) - 3.0 * self.data.qvel[self.vadr]
            self.data.ctrl[:] = np.clip(torque, self.limits[:, 0], self.limits[:, 1])
            self.data.ctrl[self.weak_motor] *= self.motor_strength()
            mujoco.mj_step(self.model, self.data)
            failed = failed or self.metrics()["fallen"]
            if failed:
                break
        self.steps += 1
        info = self.metrics()
        info["fallen"] = failed
        tilt = np.deg2rad(info["tilt_deg"])
        reward = (1.0 - 5.0 * tilt**2 - 10.0 * (info["height_m"] - 0.27)**2
                  - 2.0 * float(self.data.qpos[:2] @ self.data.qpos[:2])
                  - self.residual_penalty * float(action @ action)
                  - 0.02 * float((action - self.previous) @ (action - self.previous)))
        if failed:
            reward -= 10.0
        self.previous = action.copy()
        return self._observation(), float(reward), failed, self.steps >= self.horizon, info
