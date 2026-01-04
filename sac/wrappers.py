import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ObsWrapper(gym.ObservationWrapper):
    """
    Flattens HighwayEnv's (vehicles_count, features) observation into 1D
    and appends 2 goal scalars: [lane_delta_to_target, target_lane_norm].
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        NUM_ADDED_FEATURES = 2 # so for lane_distance_to_target and target_lane_norm
        base_shape = self.observation_space.shape  # e.g. (5, 5)
        base_dim = int(np.prod(base_shape))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_dim + NUM_ADDED_FEATURES,),
            dtype=np.float32
        )

    def observation(self, obs):
        obs_flat = np.asarray(obs, dtype=np.float32).ravel()
        obs_flat = np.nan_to_num(obs_flat, nan=0.0, posinf=0.0, neginf=0.0)

        if not np.isfinite(obs_flat).all():
            raise ValueError("Non-finite obs detected")
        
        base_env = self.env.unwrapped
        current_lane = int(base_env.vehicle.lane_index[2])
        target_lane = int(getattr(base_env, "target_lane", current_lane))

        lanes_count = int(base_env.config.get("lanes_count", 4))
        denom = max(lanes_count - 1, 1)

        lane_delta = float(np.clip((target_lane - current_lane) / denom, -1.0, 1.0))
        target_norm = float(np.clip(2.0 * (target_lane / denom) - 1.0, -1.0, 1.0))

        extra = np.array([lane_delta, target_norm], dtype=np.float32)
        return np.concatenate([obs_flat, extra], axis=0).astype(np.float32, copy=False)
