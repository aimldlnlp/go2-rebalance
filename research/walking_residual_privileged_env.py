"""Frozen walking dynamics with twelve simulator-known motor strengths appended."""
import numpy as np
from gymnasium import spaces
from walking_residual_height_env import (
    WalkingResidualEnv as OriginalEnv, CONDITIONS, RESIDUAL_SCALE,
    REWARD_WEIGHTS, HEIGHT_REFERENCE, OBSERVATION_FIELDS as ORIGINAL_FIELDS,
)

OBSERVATION_FIELDS={**ORIGINAL_FIELDS,'actual_motor_strength':[60,72]}


class WalkingResidualEnv(OriginalEnv):
    def __init__(self, condition='mixed', record_trace=False):
        super().__init__(condition=condition,record_trace=record_trace)
        self.observation_space=spaces.Box(
            np.r_[self.observation_space.low,np.full(12,.1)].astype(np.float32),
            np.r_[self.observation_space.high,np.ones(12)].astype(np.float32),dtype=np.float32)

    def _observation(self):
        strength=np.ones(12,dtype=np.float32)
        if self.onset is not None:
            strength[self.weak]=.1
        return np.r_[super()._observation(),strength].astype(np.float32)
