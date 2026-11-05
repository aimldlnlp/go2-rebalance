"""Frozen walking dynamics with only the height reward reference revised."""
from walking_residual_env import (
    CONDITIONS, OBSERVATION_FIELDS, RESIDUAL_SCALE, REWARD_WEIGHTS,
    WalkingResidualEnv as OriginalEnv,
)


HEIGHT_REFERENCE = .338


class WalkingResidualEnv(OriginalEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if 'height_error' in info['reward_terms']:
            info['reward_terms']['height_error'] = REWARD_WEIGHTS['height_error'] * (
                self.metrics['height'] - HEIGHT_REFERENCE)**2
            reward = float(sum(info['reward_terms'].values()))
        return obs, reward, terminated, truncated, info
